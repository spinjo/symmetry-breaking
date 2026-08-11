import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from experiments.distributed import gather_concat
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.experiment import TaggingExperiment, get_rej
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class JetSetTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.cfg.plotting.roc and not self.cfg.plotting.score
        self.class_names = [
            "lightjet",
            "charmjet",
            "bottomjet",
            "taujet",
        ]
        self.num_outputs = len(self.class_names)

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_jetset/fourmomenta.yaml"
        elif self.cfg.data.features == "ipsig":
            self.extra_scalars = 2
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_jetset/ipsig.yaml"
        elif self.cfg.data.features == "ip":
            self.extra_scalars = 4
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_jetset/ip.yaml"
        elif self.cfg.data.features == "ipkin":
            self.extra_scalars = 6
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_jetset/ipkin.yaml"
        elif self.cfg.data.features == "all":
            self.extra_scalars = 19
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_jetset/all.yaml"
        else:
            raise ValueError(f"Input feature option {self.cfg.data.features} not implemented")

    def _init_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "large", "test": "medium", "val": "small"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {label: frange[1] - frange[0] for label, frange in files_range.items()}
        for label, n in self.num_files.items():
            assert n >= self.world_size, (
                f"{label}: {n} files per class is less than world_size={self.world_size}; "
                "increase the file range or reduce world_size"
            )
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{label}:{path}/mc-flavtag-ttbar-{folder[label]}_{str(i).zfill(4)}.root"
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)
            file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}
            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            fraction_of_file = self.cfg.data.fraction_of_file if label == "train" else 1
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.config,
                for_training=for_training[label],
                extra_selection=self.cfg.data.extra_selection,
                remake_weights=not self.cfg.data.not_remake_weights,
                load_range_and_fraction=((0, fraction_of_file), 1, 1),
                file_fraction=1,
                fetch_by_files=self.cfg.data.fetch_by_files,
                fetch_step=self.cfg.data.fetch_step,
                infinity_mode=self.cfg.data.steps_per_epoch is not None,
                in_memory=self.cfg.data.in_memory,
                name=label,
                events_per_file=self.cfg.data.events_per_file,
                async_load=self.cfg.data.async_load,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.data.num_workers > 0
            and self.cfg.data.steps_per_epoch is not None,
        }
        # cap by per-rank file count: with external rank sharding each rank holds
        # only num_files // world_size files per class
        num_workers = {
            label: min(self.cfg.data.num_workers, self.num_files[label] // self.world_size)
            for label in ["train", "test", "val"]
        }

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["train"],
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["val"],
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=False,
            num_workers=num_workers["test"],
            **self.loader_kwargs,
        )

        self._record_train_size()
        self.init_standardization()

    @torch.inference_mode()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label)
            labels_predict.append(y_pred.float())

        labels_true = gather_concat(torch.cat(labels_true)).cpu()
        labels_predict = gather_concat(torch.cat(labels_predict)).cpu()
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # ce loss
        metrics["loss"] = torch.nn.functional.cross_entropy(labels_predict, labels_true).item()
        if mode == "eval":
            LOGGER.info(f"CELoss on {title} dataset: {metrics['loss']:.4f}")
        labels_true, labels_predict = (
            labels_true.numpy(),
            torch.softmax(labels_predict, dim=1).numpy(),
        )

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, labels_predict.argmax(1))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset:\t{metrics['accuracy']:.4f}")

        # auc and roc (fpr = epsB, tpr = epsS)
        metrics["auc_ovo"] = roc_auc_score(
            labels_true, labels_predict, multi_class="ovo", average="macro"
        )  # unweighted mean of AUCs across classes
        if mode == "eval":
            LOGGER.info(f"The ovo mean AUC is\t\t{metrics['auc_ovo']:.5f}")

        metrics_json = {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "auc_ovo": metrics["auc_ovo"],
        }

        if mode == "eval":
            LOGGER.info("### Evaluating light jets vs others (unweighted) ###")
        class_rej_list = [None, 0.6, 0.6, 0.6]
        for i in range(1, len(self.class_names)):
            labels_predict_class = labels_predict[(labels_true == 0) | (labels_true == i)]
            labels_true_class = labels_true[(labels_true == 0) | (labels_true == i)]
            labels_predict_class = labels_predict_class[:, [0, i]]

            denom = labels_predict_class[:, 0] + labels_predict_class[:, 1]
            predict_score = labels_predict_class[:, 1] / np.clip(denom, a_min=1e-10, a_max=None)

            fpr, tpr, _ = roc_curve(labels_true_class == i, predict_score)

            rej_string = str(class_rej_list[i]).replace(".", "")
            metrics[f"rej{rej_string}_{i}"] = get_rej(class_rej_list[i], tpr, fpr)
            metrics_json[f"rej{rej_string}_{self.class_names[i]}"] = metrics[f"rej{rej_string}_{i}"]
            if mode == "eval":
                LOGGER.info(
                    f"Rejection rate for class {self.class_names[i]:>10} on {title} dataset:{metrics[f'rej{rej_string}_{i}']:>5.0f} (epsS={class_rej_list[i]})"
                )

        # evaluating with class weights
        # (label, class_idx, weight)
        # class weights follow https://arxiv.org/pdf/2510.24066, not https://arxiv.org/pdf/2505.19689
        eval_classes_btag = [("cjets", 1, 0.2), ("ujets", 0, 0.79), ("taujets", 3, 0.01)]
        eval_classes_ctag = [("bjets", 2, 0.2), ("ujets", 0, 0.79), ("taujets", 3, 0.01)]
        if mode == "eval":
            LOGGER.info("### Evaluating bottom jets vs others (weighted) ###")
        metrics_b_sig = self._evaluate_single_with_weights(
            labels_true, labels_predict, 2, eval_classes_btag, mode=mode
        )
        metrics.update(metrics_b_sig)
        metrics_json.update(metrics_b_sig)

        if mode == "eval":
            LOGGER.info("### Evaluating charm jets vs others (weighted) ###")
        metrics_c_sig = self._evaluate_single_with_weights(
            labels_true, labels_predict, 1, eval_classes_ctag, mode=mode
        )
        metrics.update(metrics_c_sig)
        metrics_json.update(metrics_c_sig)

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if "labels" in key:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if self.cfg.save and mode == "eval" and title == "test":
            metrics_json = {k: float(f"{v:.6g}") for k, v in metrics_json.items()}
            self._add_run_metadata(metrics_json)
            filename = os.path.join(self.cfg.run_dir, f"results_{title}_{self.cfg.run_idx}.json")
            with open(filename, "w") as file:
                json.dump(metrics_json, file, indent=2)
        return metrics

    def _evaluate_single_with_weights(
        self, labels_true, labels_predict, idx_sig, eval_classes, mode="eval"
    ):
        metrics_with_weights = {}
        denom_class_labels = [eval_classes[i][1] for i in range(len(eval_classes))]

        labels_num = np.clip(labels_predict[:, idx_sig], a_min=1e-10, a_max=None)
        labels_denom = np.clip(
            eval_classes[0][2] * labels_predict[:, denom_class_labels[0]]
            + eval_classes[1][2] * labels_predict[:, denom_class_labels[1]]
            + eval_classes[2][2] * labels_predict[:, denom_class_labels[2]],
            a_min=1e-10,
            a_max=None,
        )
        labels_predict = np.log(labels_num / labels_denom)

        class_rej_list = [0.3] if idx_sig == 1 else [0.7]
        for n, i in enumerate(denom_class_labels):
            mask_class = (labels_true == i) | (labels_true == idx_sig)
            labels_true_class = labels_true[mask_class]
            labels_predict_class = labels_predict[mask_class]
            fpr, tpr, _ = roc_curve(labels_true_class == idx_sig, labels_predict_class)

            rej_strings = []
            for rej in class_rej_list:
                rej_string = (
                    f"{str(rej).replace('.', '')}_{self.class_names[idx_sig]}_{eval_classes[n][0]}"
                )
                rej_strings.append(rej_string)
                metrics_with_weights[f"rej{rej_string}"] = get_rej(rej, tpr, fpr)
            if mode == "eval":
                rej_report = "".join(
                    f"{metrics_with_weights[f'rej{rs}']:>5.0f} (epsS={rej})"
                    for rs, rej in zip(rej_strings, class_rej_list, strict=True)
                )
                LOGGER.info(
                    f"Rejection rate for class {self.class_names[i]:>10} on test dataset:{rej_report}"
                )
        return metrics_with_weights

    def _extract_batch(self, batch):
        fourmomenta = batch[0]["pf_vectors"].transpose(1, 2).to(self.device, self.momentum_dtype)
        if self.cfg.data.features == "fourmomenta":
            scalars = torch.empty(
                fourmomenta.shape[0],
                fourmomenta.shape[1],
                0,
                device=fourmomenta.device,
                dtype=self.dtype,
            )
        else:
            scalars = batch[0]["pf_features"].transpose(1, 2).to(self.device, self.dtype)
            if self.cfg.data.tanh_scalars:
                scalars = torch.tanh(scalars)
        label = batch[1]["_label_"].to(self.device, torch.long)
        weights = torch.ones_like(label)
        return fourmomenta, scalars, label, weights
