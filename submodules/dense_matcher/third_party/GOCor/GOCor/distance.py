# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceMap(nn.Module):
    """DistanceMap. Creates distance map (euclidian distance) from the center point. Then computes the bin values
    for each of the pixel (corresponding to the distance). For each pixel, the bin values correspond
    to a histogram of where the distance is (the bin values for each pixel must sum to 1).
    """
    def __init__(self, num_bins, bin_displacement=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.bin_displacement = bin_displacement

    def forward(self, center, output_sz):
        """
        args:
            center: torch tensor with (y,x) center position
            output_sz: size of output
        output:
            bin_val: distance map tensor
        """

        center = center.view(-1,2)

        bin_centers = torch.arange(self.num_bins, dtype=torch.float32, device=center.device).view(1, -1, 1, 1)

        k0 = torch.arange(output_sz[0], dtype=torch.float32, device=center.device).view(1,1,-1,1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32, device=center.device).view(1,1,1,-1)

        d0 = k0 - center[:, 0].view(-1, 1, 1, 1)
        d1 = k1 - center[:, 1].view(-1, 1, 1 ,1)

        dist = torch.sqrt(d0*d0 + d1*d1)
        bin_diff = dist / self.bin_displacement - bin_centers

        bin_val = torch.cat((F.relu(1.0 - torch.abs(bin_diff[:, :-1, :, :]), inplace=True),
                             (1.0 + bin_diff[:, -1:, :, :]).clamp(0, 1)), dim=1)

        return bin_val  # shape is (1, num_bins, output_sz[0], output_sz[1])


