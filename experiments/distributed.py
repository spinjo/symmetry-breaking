import torch
import torch.distributed as dist


def _initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """Average ``tensor`` across ranks in-place. No-op if not distributed."""
    if _initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def total_size_across_ranks(local_size: int, device: torch.device) -> int:
    """Sum a per-rank integer count across all ranks. No-op if not distributed.

    Use this for rank-sharded datasets where per-rank sizes may differ
    (e.g. when num_files % world_size != 0): multiplying by world_size would
    be wrong because rank 0 holds more files than other ranks.
    """
    if not _initialized():
        return local_size
    t = torch.tensor(local_size, device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def gather_concat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather a tensor across ranks and concatenate along ``dim``.

    Per-rank length along ``dim`` may differ (e.g. partial last batch);
    other dimensions must match. The result is identical on every rank.
    """
    if not _initialized():
        return tensor

    world_size = dist.get_world_size()
    device = tensor.device

    # gather sizes -> pad to global max -> all_gather -> trim padding
    local_len = torch.tensor([tensor.size(dim)], device=device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    lens = [int(x.item()) for x in lens]
    max_len = max(lens)

    if tensor.size(dim) < max_len:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_len - tensor.size(dim)
        padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
    else:
        padded = tensor.contiguous()

    bufs = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(bufs, padded)
    return torch.cat([buf.narrow(dim, 0, ln) for buf, ln in zip(bufs, lens, strict=True)], dim=dim)
