import datetime
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.tagging.atlastopexperiment import ATLASTopExperiment
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.jetsetexperiment import JetSetTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment

EXPERIMENTS = {
    "toptagging": TopTaggingExperiment,
    "toptagxl": TopTagXLExperiment,
    "jetclass": JetClassTaggingExperiment,
    "atlastop": ATLASTopExperiment,
    "jetset": JetSetTaggingExperiment,
}


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    if cfg.exp_type not in EXPERIMENTS:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    use_cuda = cfg.gpu and torch.cuda.is_available()
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
        )

    if use_cuda:
        torch.cuda.set_device(local_rank)

    try:
        exp = EXPERIMENTS[cfg.exp_type](cfg, rank, world_size, local_rank)
        exp()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    # Python 3.14 switched the default start method to forkserver, which inherits a
    # CUDA-tainted parent here (started lazily, after cuda.set_device) and segfaults.
    mp.set_start_method("fork", force=True)
    main()
