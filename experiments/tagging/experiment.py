import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from experiments.base_experiment import BaseExperiment
from experiments.distributed import gather_concat, total_size_across_ranks
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.embedding import embed_tagging_data, get_num_tagging_features
from experiments.tagging.plots import plot_mixer


def get_rej(epsS, tpr, fpr):
    """1/epsB at fixed epsS, picking the first ROC point strictly above epsS."""
    assert (tpr > epsS).any(), f"ROC never reaches tpr>{epsS}"
    return 1 / fpr[np.argmax(tpr > epsS)]


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments
    """

    def init_physics(self):
        # mirror hydra "model" choice into cfg to make it persistent
        if HydraConfig.initialized():
            model_name = HydraConfig.get().runtime.choices.get("model")
            if model_name is not None:
                with open_dict(self.cfg.model):
                    self.cfg.model.model_name = model_name

        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        self.momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else torch.float32

        self.cfg.model.out_channels = self.num_outputs
        if modelname in [
            "LGATr",
            "LGATrSlim",
            "LorentzNet",
            "PELICAN",
        ]:
            # Lorentz-equivariance by internal representations
            in_s_channels = self.extra_scalars
            in_s_channels += get_num_tagging_features(
                tagging_features=self.cfg.data.tagging_features
            )

            self.cfg.model.units = self.cfg.data.units

            if modelname in ["LGATr", "LGATrSlim"]:
                self.cfg.model.net.in_s_channels = 0 if self.cfg.model.mean_aggregation else 1
                self.cfg.model.net.in_s_channels += in_s_channels
                if self.cfg.model.rescale:
                    self.cfg.model.net.in_s_channels += 1
            elif modelname == "LorentzNet":
                self.cfg.model.net.n_scalar = in_s_channels
            elif modelname == "PELICAN":
                self.cfg.model.net.in_channels_rank1 = in_s_channels

        elif modelname in [
            "Transformer",
            "ParticleTransformer",
            "ParticleNet",
            "MIParticleTransformer",
            "PET2",
            "SaltModel",
        ]:
            # Non-equivariant or canonicalization
            self.cfg.model.in_channels = 7 + self.extra_scalars

            if modelname == "Transformer":
                self.cfg.model.in_channels += 0 if self.cfg.model.mean_aggregation else 1
            elif modelname == "ParticleNet":
                self.cfg.model.net.hidden_reps_list[0] = f"{self.cfg.model.in_channels}x0n"
            elif modelname == "SaltModel":
                self.cfg.model.net.tasks.modules[0].class_names = [
                    f"c{i}" for i in range(self.num_outputs)
                ]
                self.cfg.model.net.encoder.attn_type = (
                    "torch-meff" if self.cfg.model.zeropad else "flash-varlen"
                )
            elif modelname == "PET2":
                assert self.cfg.data.tagging_features == "all", (
                    "PET2 requires tagging_features=all for internal operations"
                )

            # different treatments in LLoCa and non-equivariant networks
            if "equivectors" in self.cfg.model.framesnet:
                # decide which entries to use for the framesnet
                num_tagging_features = get_num_tagging_features(
                    tagging_features=self.cfg.data.tagging_features
                )
                self.cfg.model.framesnet.equivectors.num_scalars = self.extra_scalars
                self.cfg.model.framesnet.equivectors.num_scalars += num_tagging_features
                self.cfg.model.framesnet.mass_reg = self.cfg.data.mass_reg
            else:
                # turn off spurions
                self.cfg.data.beam_reference = None
                self.cfg.data.add_time_reference = False

                # not allowed, because the network is not Lorentz-equivariant
                if self.cfg.data.canonicalize == "rest":
                    self.cfg.data.canonicalize = "beam_y"
        else:
            raise NotImplementedError(f"Model {modelname} not implemented")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

        self._record_train_size()
        self.init_standardization()

    def _record_train_size(self):
        n_train = len(self.data_train)
        if isinstance(self.data_train, torch.utils.data.IterableDataset):
            # rank-sharded IterableDataset: per-rank lengths may differ when
            # num_files % world_size != 0, so sum instead of multiplying.
            n_train = total_size_across_ranks(n_train, self.device)
        self.train_size = n_train
        LOGGER.info(f"Training dataset has {n_train} elements")

    def init_standardization(self):
        if hasattr(self._model, "init_standardization"):
            batch = next(iter(self.train_loader))
            fourmomenta, scalars, _, _ = self._extract_batch(batch)
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                self.cfg.data,
            )
            self._model.init_standardization(
                embedding[0], mask=embedding[-1], is_spurion=embedding[3]
            )
            # each rank sees a different first batch, so broadcast rank 0's buffers
            if self.world_size > 1:
                for buf in self._model.buffers():
                    dist.broadcast(buf, src=0)

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            self.results[set_label] = self._evaluate_single(
                loader_dict[set_label], set_label, mode="eval"
            )

    def plot(self):
        if not self.is_master:
            return
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        title = type(self._model.net).__name__
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack((self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1)
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = torch.stack(self.grad_norm_train).cpu()
            plot_dict["grad_norm_frames"] = torch.stack(self.grad_norm_frames).cpu()
            plot_dict["grad_norm_net"] = torch.stack(self.grad_norm_net).cpu()
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker, _, weights = self._get_ypred_and_label(batch)
        loss = torch.mean(weights * self.loss(y_pred, label))

        metrics = tracker
        return loss, metrics

    def _get_ypred_and_label(self, batch):
        fourmomenta, scalars, label, weights = self._extract_batch(batch)
        embedding_list = embed_tagging_data(
            fourmomenta,
            scalars,
            self.cfg.data,
        )
        y_pred, tracker, frames = self.model(*embedding_list)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]
        return y_pred, label, tracker, frames, weights

    def _init_metrics(self):
        return {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }

    def _add_run_metadata(self, metrics_json):
        metrics_json["model_name"] = self.cfg.model.get("model_name")
        metrics_json["model_size"] = self.cfg.model.net.get("size")
        metrics_json["train_size"] = self.train_size

    def init_data(self):
        raise NotImplementedError

    def _evaluate_single(self, loader, title, mode, step=None):
        raise NotImplementedError

    def _init_loss(self):
        raise NotImplementedError

    def _extract_batch(self, batch):
        # it should return (fourmomenta, scalars, labels, weights)
        raise NotImplementedError


class BinaryTaggingExperiment(TaggingExperiment):
    @torch.inference_mode()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            # IterableDataset.__len__ is per-rank (rank-sharded file_dict);
            # map-style dataset.__len__ is global. Show the global total either way.
            n = len(loader.dataset)
            if isinstance(loader.dataset, torch.utils.data.IterableDataset):
                n = total_size_across_ranks(n, self.device)
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{n} elements, batchsize {loader.batch_size * self.world_size} ###"
            )
        metrics = {}

        labels_true, labels_predict, weights = [], [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _, w = self._get_ypred_and_label(batch)
            labels_true.append(label.float())
            labels_predict.append(y_pred.float())
            weights.append(w.float())
        labels_true = gather_concat(torch.cat(labels_true)).cpu()
        labels_predict = gather_concat(torch.cat(labels_predict)).cpu()
        weights = gather_concat(torch.cat(weights)).cpu()

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )
            metrics["weights"] = weights

        # bce loss (matches the training objective: mean of weight*BCE over events)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            labels_predict, labels_true, reduction="none"
        )
        metrics["loss"] = (weights * bce).mean().item()
        if mode == "eval":
            LOGGER.info(f"BCELoss on {title} dataset: {metrics['loss']:.6f}")
        labels_predict = torch.nn.functional.sigmoid(labels_predict)
        labels_true, labels_predict, weights = (
            labels_true.numpy(),
            labels_predict.numpy(),
            weights.numpy(),
        )

        # accuracy
        metrics["accuracy"] = accuracy_score(
            labels_true, np.round(labels_predict), sample_weight=weights
        )
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.6f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict, sample_weight=weights)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict, sample_weight=weights)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.6f}")

        metrics["rej03"] = get_rej(0.3, tpr, fpr)
        metrics["rej05"] = get_rej(0.5, tpr, fpr)
        metrics["rej08"] = get_rej(0.8, tpr, fpr)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr", "weights"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if self.cfg.save and mode == "eval" and title == "test":
            metrics_json = {
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "rej03": metrics["rej03"],
                "rej05": metrics["rej05"],
                "rej08": metrics["rej08"],
            }
            metrics_json = {k: float(f"{v:.6g}") for k, v in metrics_json.items()}
            self._add_run_metadata(metrics_json)
            filename = os.path.join(self.cfg.run_dir, f"results_{title}_{self.cfg.run_idx}.json")
            with open(filename, "w") as file:
                json.dump(metrics_json, file, indent=2)
        return metrics

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")


class TopTaggingExperiment(BinaryTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 0

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz")
        LOGGER.info(f"Creating dataset from {data_path}")
        t0 = time.time()
        file = np.load(data_path)

        def get_dataset(label):
            fourmomenta = file[f"kinematics_{label}"]
            labels = file[f"labels_{label}"]
            momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else self.dtype
            fourmomenta = torch.tensor(fourmomenta, dtype=momentum_dtype)
            scalars = torch.zeros(*fourmomenta.shape[:-1], 0, dtype=self.dtype)
            labels = torch.tensor(labels, dtype=self.dtype)
            return torch.utils.data.TensorDataset(fourmomenta, scalars, labels)

        self.data_train = get_dataset("train")
        self.data_test = get_dataset("test")
        self.data_val = get_dataset("val")
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _extract_batch(self, batch):
        fourmomenta = batch[0].to(self.device)
        scalars = batch[1].to(self.device)
        label = batch[2].to(self.device)
        weights = torch.ones_like(label)
        return fourmomenta, scalars, label, weights
