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


class SeparableConv4d(nn.Module):
    """
    4D convolution, implemented as 2 2D convolutions applied sequentially
    args:
        kernel_size: kernel size for the 2 2D convolutions
        input_dim: channel dimension of input channel, must be 1
        inter_dim: channel dimension after first 2D convolution
        output_dim: channel dimension after second 2D convolution (final output)
        bias:
        padding:
        permute_back_output: permute output
    """
    def __init__(self, kernel_size=3, input_dim=1, inter_dim=1, output_dim=1, bias=True, padding=None,
                 permute_back_output=True):
        super().__init__()

        assert input_dim==1

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.weight1 = nn.Parameter(torch.zeros(inter_dim, input_dim, *kernel_size))
        self.weight2 = nn.Parameter(torch.zeros(output_dim, inter_dim, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

        self.padding = [k//2 for k in kernel_size] if padding is None else padding

        self.permute_back_output = permute_back_output

    def forward(self, x, transpose=False):
        """
        args:
            x: tensor on which to apply the 4D volution,
               dimension must be (num_sequences, H2, W2, output_dim, H1, W1) if transpose = True
               or (num_sequences, H2, W2,  H1, W1) if transpose = False
            transpose: apply transpose 4D convolution
        :return:
            x3: tensor after 4D conv or transposed 4D conv
        """
        input_dim = 1
        output_dim, inter_dim = self.weight2.shape[:2]

        if transpose:
            # Expect 6D input
            # x is (b, H2, W2, output_dim, H1, W1)
            assert x.dim() == 6

            if self.permute_back_output:
                x = x.permute(0, 4, 5, 3, 1, 2)
                # x is (b, H1, W1, output_dim, H2, W2)

            batch_size = x.shape[0]
            sz1 = x.shape[1:3]
            sz2 = x.shape[-2:]

            # inverse convolution
            x2 = F.conv_transpose2d(x.reshape(-1, output_dim, *sz2), self.weight2, padding=self.padding)
            x2 = x2.reshape(batch_size, sz1[0]*sz1[1], inter_dim, sz2[0]*sz2[1]).permute(0,3,2,1)

            x3 = F.conv_transpose2d(x2.reshape(-1, inter_dim, *sz1), self.weight1, padding=self.padding)
            # ends up with b, sz2*, input_dim, sz1*

            return x3.reshape(batch_size, *sz2, *sz1)

        # Expect 5D input
        # shape is b, H2, W2, H1, W1
        assert x.dim() == 5

        batch_size = x.shape[0]
        sz2 = x.shape[1:3]  # (H2, W2)
        sz1 = x.shape[-2:]  # (H1, W1)

        # reshape (b*H2*W2, 1, H1, W1)
        x2 = F.conv2d(x.reshape(-1, input_dim, *sz1), self.weight1, padding=self.padding)
        x2 = x2.reshape(batch_size, sz2[0]*sz2[1], inter_dim, sz1[0]*sz1[1]).permute(0, 3, 2, 1)
        # shape is b, H2*W2, inter_dim, H1*W1 then permute

        # reshape (b*W1*H1, inter_dim, H2, W2)
        x3 = F.conv2d(x2.reshape(-1, inter_dim, *sz2), self.weight2, bias=self.bias, padding=self.padding)
        x3 = x3.reshape(batch_size, *sz1, output_dim, *sz2)  # reshape (b, H1, W1, output_dim, H2, W2)

        if self.permute_back_output:
            x3 = x3.permute(0, 4, 5, 3, 1, 2).contiguous()
            # shape is b, H2, W2, output_dim, H1, W1

        return x3
