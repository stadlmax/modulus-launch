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

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional


class Constants(BaseModel):
    """GraphCast constants"""

    processor_layers: int = 8
    hidden_dim: int = 256
    segments: int = 1
    force_single_checkpoint: bool = False
    checkpoint_encoder: bool = True
    checkpoint_processor: bool = True
    checkpoint_decoder: bool = True
    force_single_checkpoint_finetune: bool = False
    checkpoint_encoder_finetune: bool = True
    checkpoint_processor_finetune: bool = True
    checkpoint_decoder_finetune: bool = True
    concat_trick: bool = True
    cugraphops_encoder: bool = False # True
    cugraphops_processor: bool = False # True
    cugraphops_decoder: bool = False # True
    recompute_activation: bool = True
    wb_mode: str = "disabled"
    dataset_path: str = "datasets/ngc_era5_data"
    static_dataset_path: str = "datasets/static"
    latlon_res: Tuple[int, int] = (721, 1440)
    num_workers: int = 0 # 8
    num_channels: int = 34
    num_channels_val: int = 34
    num_val_steps: int = 8
    num_val_spy: int = 1  # SPY: Samples Per Year
    grad_clip_norm: Optional[float] = 32.0
    jit: bool = False
    amp: bool = False
    amp_dtype: str = "bfloat16"
    full_bf16: bool = True
    watch_model: bool = False
    lr: float = 1e-3
    lr_step3: float = 3e-7
    num_iters_step1 = 100 #1000
    num_iters_step2 = 100 #299000
    num_iters_step3 = 0 #11000
    step_change_freq = 1000
    save_freq: int = 500
    val_freq: int = 1000
    ckpt_path: str = "tmp/checkpoints_34var_1"
    val_dir: str = "tmp/validation_34var_1"
    ckpt_name: str = "model_34var.pt"
    pyt_profiler: bool = False
    profile: bool = False
    profile_range: Tuple = (90, 110)
    icospheres_path: str = os.path.join(
        Path(__file__).parent.resolve(), "icospheres.json"
    )
    partition_size: int = 1
