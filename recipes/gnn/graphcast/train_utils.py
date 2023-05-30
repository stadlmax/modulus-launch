# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from typing import Optional, List


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters of.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_process_groups(partition_size: int) -> List[dist.ProcessGroup]:
    world_size = dist.get_world_size()
    assert world_size > 1, "distributed training not initialized"
    assert world_size > partition_size, "world_size must be larger than requested number of partitions"
    assert world_size % partition_size == 0, "partition_size must divide world_size evenly"

    num_partitions = world_size // partition_size
    partition_groups = [None] * num_partitions

    for p in range(num_partitions):
        tmp = list(range(p * partition_size, (p + 1) * partition_size))
        partition_groups[p] = dist.new_group(
            ranks=tmp, backend='nccl'
        )

    return partition_groups


def custom_allreduce_fut(
        process_group: dist.ProcessGroup, 
        tensor: torch.Tensor, 
        divisor: Optional[int] = None
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    if divisor is not None:
        tensor.div_(divisor)
    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True, op=dist.ReduceOp.SUM)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )
