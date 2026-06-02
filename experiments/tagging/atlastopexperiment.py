import json
import os
import time
from glob import glob

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from experiments.distributed import gather_concat, total_size_across_ranks
from experiments.logger import LOGGER
from experiments.tagging.experiment import BinaryTaggingExperiment, get_rej
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist

ATLAS_SYST_NAMES = (
    "angular",
    "bias",
    "cer",
    "cluster",
    "cpos",
    "dipole",
    "esdown",
    "esup",
    "string",
    "teg",
    "tej",
    "tfj",
    "tfl",
    "ttbar_herwig",
    "ttbar_pythia",
)
ATLAS_BKG_ONLY_SYSTS = ("angular", "cluster", "dipole", "string")
ATLAS_SIG_ONLY_SYSTS = ("ttbar_herwig", "ttbar_pythia")
ATLAS_SHOWER_COLS = {"ISRx2": 4, "FSRx2": 6, "FSRxp5": 7, "ISRxp5": 9}


def _concat_into(target, source):
    for key in ("labels_true", "labels_predict"):
        target[key] = np.concatenate([target[key], source[key]], axis=0)


def _compute_metrics(labels_true, labels_predict, sample_weight=None):
    fpr, tpr, _ = roc_curve(labels_true, labels_predict, sample_weight=sample_weight)
    rej05 = get_rej(0.5, tpr, fpr)
    auc = roc_auc_score(labels_true, labels_predict, sample_weight=sample_weight)
    return rej05, auc


def _safe_max(*xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _safe_quad(*xs):
    if any(x is None for x in xs):
        return None
    return float(np.sqrt(sum(x**2 for x in xs)))


class ATLASTopExperiment(BinaryTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.eval_systs = self.cfg.data.eval_systs
        if self.eval_systs:
            self.systs_set = self.cfg.data.systs_set
            self.syst_set_names = ["all", *ATLAS_SYST_NAMES]
            assert all(syst in self.syst_set_names for syst in self.systs_set)
            if self.systs_set[0] == "all":
                active = ATLAS_SYST_NAMES
            else:
                active = self.systs_set
            self.syst_folders = {syst: f"{syst}" for syst in active}
            self.syst_datasets = {syst: None for syst in active}

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.config = {
                "train": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta.yaml",
                "val": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta.yaml",
                "test": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_test.yaml",
                "syst": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_noweights.yaml",
                "onlyqcd": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_onlyqcd.yaml",
                "onlytop": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_onlytop.yaml",
            }
        else:
            raise ValueError(f"Input feature option {self.cfg.data.features} not implemented")

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_nominal", "test": "test_nominal", "val": "val_nominal"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {label: frange[1] - frange[0] for label, frange in files_range.items()}
        for label, n in self.num_files.items():
            assert n >= self.world_size, (
                f"{label}: {n} files is less than world_size={self.world_size}; "
                "increase the file range or reduce world_size"
            )
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{folder[label]}:{path}/{folder[label]}_{str(i).zfill(3)}.root"
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)
            file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            fraction_of_file = self.cfg.data.fraction_of_file if label == "train" else 1
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.config[label],
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

        if self.eval_systs:
            for syst in self.syst_folders.keys():
                path = os.path.join(self.cfg.data.data_dir, self.syst_folders[syst])
                flist = glob(f"{path}/{self.syst_folders[syst]}_*.root")
                self.num_files[syst] = len(flist)
                file_dict, _ = to_filelist(flist)
                file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

                LOGGER.info(f"Using {len(flist)} files for syst {syst} from {path}")
                self.syst_datasets[syst] = SimpleIterDataset(
                    file_dict,
                    self.cfg.data.config["syst"],
                    for_training=False,
                    extra_selection=self.cfg.data.extra_selection,
                    remake_weights=not self.cfg.data.not_remake_weights,
                    load_range_and_fraction=((0, 1), 1, 1),
                    file_fraction=1,
                    fetch_by_files=self.cfg.data.fetch_by_files,
                    fetch_step=self.cfg.data.fetch_step,
                    infinity_mode=self.cfg.data.steps_per_epoch is not None,
                    in_memory=self.cfg.data.in_memory,
                    name=syst,
                    events_per_file=self.cfg.data.events_per_file,
                    async_load=self.cfg.data.async_load,
                )

            additional_datasets = ["onlyqcd", "onlytop"]
            for label in additional_datasets:
                path = os.path.join(self.cfg.data.data_dir, "test_nominal")
                flist = [
                    f"{label}:{path}/test_nominal_{str(i).zfill(3)}.root"
                    for i in range(*files_range["test"])
                ]
                self.num_files[label] = len(flist)
                file_dict, _ = to_filelist(flist)
                file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

                LOGGER.info(f"Using {len(flist)} files for dataset {label} from {path}")
                self.syst_datasets[label] = SimpleIterDataset(
                    file_dict,
                    self.cfg.data.config[label],
                    for_training=False,
                    extra_selection=self.cfg.data.extra_selection,
                    remake_weights=not self.cfg.data.not_remake_weights,
                    load_range_and_fraction=((0, 1), 1, 1),
                    file_fraction=1,
                    fetch_by_files=self.cfg.data.fetch_by_files,
                    fetch_step=self.cfg.data.fetch_step,
                    infinity_mode=self.cfg.data.steps_per_epoch is not None,
                    in_memory=self.cfg.data.in_memory,
                    name=label,
                    events_per_file=self.cfg.data.events_per_file,
                    async_load=self.cfg.data.async_load,
                )

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

        if self.eval_systs:
            self.syst_loaders = {
                syst: DataLoader(
                    dataset=self.syst_datasets[syst],
                    batch_size=self.cfg.evaluation.batchsize // self.world_size,
                    drop_last=False,
                    num_workers=min(
                        self.cfg.data.num_workers,
                        max(1, self.num_files[syst] // self.world_size),
                    ),
                    **self.loader_kwargs,
                )
                for syst in self.syst_datasets.keys()
            }

        self._record_train_size()
        self.init_standardization()

    def _extract_batch(self, batch):
        fourmomenta = batch[0]["pf_vectors"].transpose(1, 2).to(self.device, self.momentum_dtype)
        weights = batch[0]["ev_weights"].to(self.device, self.dtype)[..., 0, 0]
        if self.cfg.data.features == "fourmomenta":
            scalars = torch.empty(
                fourmomenta.shape[0],
                fourmomenta.shape[1],
                0,
                device=fourmomenta.device,
                dtype=self.dtype,
            )
        label = batch[1]["_label_"].to(self.device, self.dtype)
        return fourmomenta, scalars, label, weights

    def evaluate(self):
        super().evaluate()
        if not self.eval_systs:
            return

        self.model.eval()
        labels_true, labels_predict, shower_weights = [], [], []
        with torch.inference_mode():
            for batch in self.test_loader:
                y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
                labels_true.append(label.float())
                labels_predict.append(y_pred.float())
                shower_weights.append(
                    batch[0]["shower_weights"].to(self.device, self.dtype)[..., 0, :].float()
                )
        labels_true = gather_concat(torch.cat(labels_true)).cpu().numpy()
        labels_predict = (
            torch.nn.functional.sigmoid(gather_concat(torch.cat(labels_predict))).cpu().numpy()
        )
        shower_weights = gather_concat(torch.cat(shower_weights)).cpu().numpy()
        test_result = self.results.setdefault("test", {})
        test_result["labels_true"] = labels_true
        test_result["labels_predict"] = labels_predict
        test_result["shower_weights"] = shower_weights

        for syst_name, loader in self.syst_loaders.items():
            dataset_size = len(loader.dataset)
            if isinstance(loader.dataset, torch.utils.data.IterableDataset):
                dataset_size = total_size_across_ranks(dataset_size, self.device)
            LOGGER.info(
                f"### Starting to evaluate model on {syst_name} dataset with "
                f"{dataset_size} elements, batchsize {loader.batch_size * self.world_size} ###"
            )
            labels_true, labels_predict = [], []
            with torch.inference_mode():
                for batch in loader:
                    y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
                    labels_true.append(label.float())
                    labels_predict.append(y_pred.float())
            labels_true = gather_concat(torch.cat(labels_true)).cpu()
            labels_predict = gather_concat(torch.cat(labels_predict)).cpu()
            labels_predict = torch.nn.functional.sigmoid(labels_predict)
            self.results[syst_name] = {
                "labels_true": labels_true.numpy(),
                "labels_predict": labels_predict.numpy(),
            }

        if not self.is_master:
            return

        # alt-sample-only systs are class-incomplete; append nominal opposite-class jets
        for syst_name in ATLAS_BKG_ONLY_SYSTS:
            if syst_name in self.results and "onlytop" in self.results:
                _concat_into(self.results[syst_name], self.results["onlytop"])
        for syst_name in ATLAS_SIG_ONLY_SYSTS:
            if syst_name in self.results and "onlyqcd" in self.results:
                _concat_into(self.results[syst_name], self.results["onlyqcd"])

        if not self.cfg.save:
            return

        LOGGER.info("### Computing ATLAS systematics summary")
        nominal_results = self.results["test"]
        nominal_labels_true = nominal_results["labels_true"]
        nominal_labels_predict = nominal_results["labels_predict"]
        shower_weights = nominal_results["shower_weights"]
        nominal_rej05, nominal_auc = _compute_metrics(nominal_labels_true, nominal_labels_predict)
        nominal_metrics = {"rej05": nominal_rej05, "auc": nominal_auc}

        metrics_per_syst = {}
        for syst_name in ATLAS_SYST_NAMES:
            if syst_name in self.results:
                rej05, auc = _compute_metrics(
                    self.results[syst_name]["labels_true"],
                    self.results[syst_name]["labels_predict"],
                )
                metrics_per_syst[syst_name] = {"rej05": rej05, "auc": auc}

        nominal_w = shower_weights[:, 0]
        sig_mask = nominal_labels_true > 0.5
        bkg_mask = ~sig_mask
        for variation, col in ATLAS_SHOWER_COLS.items():
            ratio = shower_weights[:, col] / nominal_w
            for side, mask in (("sig", sig_mask), ("bkg", bkg_mask)):
                weights = np.ones(nominal_labels_true.shape, dtype=np.float64)
                weights[mask] = ratio[mask]
                rej05, auc = _compute_metrics(
                    nominal_labels_true, nominal_labels_predict, sample_weight=weights
                )
                metrics_per_syst[f"{side}_{variation}"] = {"rej05": rej05, "auc": auc}

        metrics_json = {}
        for metric_name in ("rej05", "auc"):
            nominal_value = nominal_metrics[metric_name]
            rel_unc = {
                syst_name: abs(syst_metrics[metric_name] - nominal_value) / nominal_value
                for syst_name, syst_metrics in metrics_per_syst.items()
            }
            pair_ratios = {}
            for numerator, denominator in (
                ("ttbar_herwig", "ttbar_pythia"),
                ("dipole", "angular"),
                ("cluster", "string"),
            ):
                if numerator in metrics_per_syst and denominator in metrics_per_syst:
                    pair_ratios[(numerator, denominator)] = abs(
                        metrics_per_syst[numerator][metric_name]
                        / metrics_per_syst[denominator][metric_name]
                        - 1
                    )

            leaf_uncs = {
                "unc_es": _safe_max(rel_unc.get("esup"), rel_unc.get("esdown")),
                "unc_cer": rel_unc.get("cer"),
                "unc_cpos": rel_unc.get("cpos"),
                "unc_eff": _safe_max(rel_unc.get("teg"), rel_unc.get("tej")),
                "unc_fake": _safe_max(rel_unc.get("tfl"), rel_unc.get("tfj")),
                "unc_bias": rel_unc.get("bias"),
                "unc_sig_model": pair_ratios.get(("ttbar_herwig", "ttbar_pythia")),
                "unc_bkg_ps": pair_ratios.get(("dipole", "angular")),
                "unc_bkg_had": pair_ratios.get(("cluster", "string")),
                "unc_sig_ISR": _safe_max(rel_unc.get("sig_ISRx2"), rel_unc.get("sig_ISRxp5")),
                "unc_sig_FSR": _safe_max(rel_unc.get("sig_FSRx2"), rel_unc.get("sig_FSRxp5")),
                "unc_bkg_ISR": _safe_max(rel_unc.get("bkg_ISRx2"), rel_unc.get("bkg_ISRxp5")),
                "unc_bkg_FSR": _safe_max(rel_unc.get("bkg_FSRx2"), rel_unc.get("bkg_FSRxp5")),
            }
            group_uncs = {
                "unc_cluster": _safe_quad(
                    leaf_uncs["unc_es"], leaf_uncs["unc_cer"], leaf_uncs["unc_cpos"]
                ),
                "unc_track": _safe_quad(
                    leaf_uncs["unc_eff"], leaf_uncs["unc_fake"], leaf_uncs["unc_bias"]
                ),
                "unc_bkg_model": _safe_quad(leaf_uncs["unc_bkg_ps"], leaf_uncs["unc_bkg_had"]),
                "unc_scale": _safe_quad(
                    leaf_uncs["unc_sig_ISR"],
                    leaf_uncs["unc_sig_FSR"],
                    leaf_uncs["unc_bkg_ISR"],
                    leaf_uncs["unc_bkg_FSR"],
                ),
            }
            unc_total = _safe_quad(
                group_uncs["unc_cluster"],
                group_uncs["unc_track"],
                leaf_uncs["unc_sig_model"],
                group_uncs["unc_bkg_model"],
                group_uncs["unc_scale"],
            )

            metric_dict = {"nominal": nominal_value}
            for syst_name, syst_metrics in metrics_per_syst.items():
                metric_dict[syst_name] = syst_metrics[metric_name]
            metric_dict.update({k: v for k, v in leaf_uncs.items() if v is not None})
            metric_dict.update({k: v for k, v in group_uncs.items() if v is not None})
            if unc_total is not None:
                metric_dict["unc_total"] = unc_total
            for key, value in metric_dict.items():
                metrics_json[f"{metric_name}_{key}"] = value

        for metric_name in ("rej05", "auc"):
            total_key = f"{metric_name}_unc_total"
            if total_key in metrics_json:
                LOGGER.info(f"{total_key} = {metrics_json[total_key]:.4f}")

        metrics_json = {k: float(f"{v:.6g}") for k, v in metrics_json.items()}
        self._add_run_metadata(metrics_json)
        filename = os.path.join(self.cfg.run_dir, f"results_test_{self.cfg.run_idx}.json")
        existing = {}
        if os.path.exists(filename):
            with open(filename) as f:
                existing = json.load(f)
        existing.update(metrics_json)
        with open(filename, "w") as f:
            json.dump(existing, f, indent=2)
