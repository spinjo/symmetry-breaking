import os
import time

import torch
from torch.utils.data import DataLoader

from experiments.logger import LOGGER
from experiments.tagging.experiment import BinaryTaggingExperiment
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class TopTagXLExperiment(BinaryTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.class_names = ["qcd", "top"]

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.config = (
                "experiments/tagging/miniweaver/configs_toptagxl/fourmomenta.yaml"
            )
        elif self.cfg.data.features == "pid":
            self.extra_scalars = 6
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_toptagxl/pid.yaml"
        elif self.cfg.data.features == "displacements":
            self.extra_scalars = 4
            self.cfg.data.config = (
                "experiments/tagging/miniweaver/configs_toptagxl/displacements.yaml"
            )
        elif self.cfg.data.features == "all":
            self.extra_scalars = 10
            self.cfg.data.config = "experiments/tagging/miniweaver/configs_toptagxl/all.yaml"
        else:
            raise ValueError(f"Input feature option {self.cfg.data.features} not implemented")

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_100M", "test": "test_25M", "val": "val_10M"}
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
                f"{classname}:{path}/{classname}_{str(i).zfill(3)}.root"
                for classname in self.class_names
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
        label = batch[1]["_label_"].to(self.device, self.dtype)
        weights = torch.ones_like(label)
        return fourmomenta, scalars, label, weights
