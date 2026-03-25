"""
Helpers for optional torch.distributed setup.
"""
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None
    device: torch.device | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed_mode(base_device: torch.device) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False, device=base_device)

    if base_device.type != "cuda":
        raise RuntimeError(
            "Distributed training is currently supported only on CUDA. "
            "Launch without torchrun on MPS or CPU."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend="nccl",
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(module, distributed: DistributedContext):
    if not distributed.enabled:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[distributed.local_rank],
        output_device=distributed.local_rank,
    )


def reduce_mean(value, distributed: DistributedContext) -> torch.Tensor:
    if not torch.is_tensor(value):
        device = distributed.device or torch.device("cpu")
        value = torch.tensor(value, device=device)

    reduced = value.detach().clone()
    if distributed.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= distributed.world_size
    return reduced


def reduce_sum(value, distributed: DistributedContext) -> torch.Tensor:
    if not torch.is_tensor(value):
        device = distributed.device or torch.device("cpu")
        value = torch.tensor(value, device=device)

    reduced = value.detach().clone()
    if distributed.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module
