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
import sys
import torch
import matplotlib.pyplot as plt

from constants import Constants
from modulus.datapipes.climate import ERA5HDF5Datapipe

C = Constants()


class Validation:
    def __init__(self, model, dtype, dist, wb):
        self.model = model
        self.dtype = dtype
        self.dist = dist
        self.wb = wb
        self.val_datapipe = ERA5HDF5Datapipe(
            data_dir=os.path.join(C.dataset_path, "test"),
            stats_dir=os.path.join(C.dataset_path, "stats"),
            channels=[i for i in range(C.num_channels)],
            num_steps=C.num_val_steps,
            batch_size=1,
            num_samples_per_year=C.num_val_spy,
            shuffle=False,
            device=self.dist.device,
            process_rank=self.dist.rank // C.partition_size,
            world_size=self.dist.world_size // C.partition_size,
            num_workers=C.num_workers,
        )
        print(f"Loaded validation datapipe of size {len(self.val_datapipe)}")

    @torch.no_grad()
    def step(self, channels=[0, 1, 2], iter=0):
        torch.cuda.nvtx.range_push("Validation")
        os.makedirs(C.val_dir, exist_ok=True)
        loss_epoch = 0

        for i, data in enumerate(self.val_datapipe):
            invar = data[0]["invar"].to(dtype=self.dtype)
            outvar = (
                data[0]["outvar"][0].to(dtype=self.dtype).to(device=self.dist.device)
            )
            invar_shape = invar.shape

            if (
                isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                and self.model.module.is_distributed
            ):
                _N, _C, _H, _W = invar.shape
                invar = invar.view(_N, _C, _H * _W)
                invar = invar.permute(2, 1, 0).view(_H * _W, -1)
                invar = self.model.module.g2m_graph.get_src_node_features_in_partition(
                    invar
                )
                invar = invar.permute(1, 0).unsqueeze(dim=0)

            pred = (
                torch.empty(outvar.shape)
                .to(dtype=self.dtype)
                .to(device=self.dist.device)
            )
            for t in range(outvar.shape[0]):
                # all ranks have to take part in forward pass
                outpred = self.model(invar)
                invar = outpred

                if (
                    isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                    and self.model.module.is_distributed
                ):
                    outpred = outpred.permute(2, 0, 1)
                    outpred = outpred.view(outpred.size(0), -1)
                    outpred = self.model.module.m2g_graph.get_global_dst_node_features(
                        outpred, get_on_all_ranks=False
                    )
                    outpred = outpred.permute(1, 0).unsqueeze(dim=0).view(invar_shape)
                    pred[t] = outpred

                else:
                    pred[t] = outpred

            if self.dist.rank == 0:
                loss_epoch += torch.mean(torch.pow(pred - outvar, 2))
                torch.cuda.nvtx.range_pop()

                pred = pred.to(torch.float32).cpu().numpy()
                outvar = outvar.to(torch.float32).cpu().numpy()

            del invar, outpred
            torch.cuda.empty_cache()

            if i == 0 and self.dist.rank == 0:
                for chan in channels:
                    plt.close("all")
                    fig, ax = plt.subplots(3, pred.shape[0], figsize=(15, 5))
                    fig.subplots_adjust(hspace=0.5, wspace=0.3)

                    for t in range(outvar.shape[0]):
                        im_pred = ax[0, t].imshow(pred[t, chan], vmin=-1.5, vmax=1.5)
                        ax[0, t].set_title(f"Prediction (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_pred, ax=ax[0, t], orientation="horizontal", pad=0.4
                        )

                        im_outvar = ax[1, t].imshow(
                            outvar[t, chan], vmin=-1.5, vmax=1.5
                        )
                        ax[1, t].set_title(f"Ground Truth (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_outvar, ax=ax[1, t], orientation="horizontal", pad=0.4
                        )

                        im_diff = ax[2, t].imshow(
                            abs(pred[t, chan] - outvar[t, chan]), vmin=0.0, vmax=0.5
                        )
                        ax[2, t].set_title(f"Abs. Diff. (t={t+1})", fontsize=10)
                        fig.colorbar(
                            im_diff, ax=ax[2, t], orientation="horizontal", pad=0.4
                        )

                    fig.savefig(
                        os.path.join(
                            C.val_dir, f"era5_validation_channel{chan}_iter{iter}.png"
                        )
                    )
                    self.wb.log({f"val_chan{chan}_iter{iter}": fig}, step=iter)

        return loss_epoch / len(self.val_datapipe)
