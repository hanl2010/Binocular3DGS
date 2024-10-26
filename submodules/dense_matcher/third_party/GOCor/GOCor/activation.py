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


def softmax_reg(x: torch.Tensor, dim, reg=None):
    """Softmax with optinal denominator regularization."""
    if reg is None:
        return torch.softmax(x, dim=dim)
    dim %= x.dim()
    if isinstance(reg, (float, int)):
        reg = x.new_tensor([reg])
    reg = reg.expand([1 if d==dim else x.shape[d] for d in range(x.dim())])
    x = torch.cat((x, reg), dim=dim)
    return torch.softmax(x, dim=dim)[[slice(-1) if d==dim else slice(None) for d in range(x.dim())]]


def logsumexp_reg(x: torch.Tensor, dim, reg=None):
    """Softmax with optinal denominator regularization."""
    if reg is None:
        return torch.logsumexp(x, dim=dim)
    dim %= x.dim()
    if isinstance(reg, (float, int)):
        reg = x.new_tensor([reg])
    reg = reg.expand([1 if d==dim else x.shape[d] for d in range(x.dim())])
    x = torch.cat((x, reg), dim=dim)
    return torch.logsumexp(x, dim=dim)


class MLU(nn.Module):
    """MLU activation
    """
    def __init__(self, min_val, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.inplace = inplace

    def forward(self, input):
        return F.elu(F.leaky_relu(input, 1/self.min_val, inplace=self.inplace), self.min_val, inplace=self.inplace)


class LeakyReluPar(nn.Module):
    """LeakyRelu parametric activation
    """

    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.abs(x) + (1.0 + a)/2.0 * x


class LeakyReluParDeriv(nn.Module):
    """Derivative of the LeakyRelu parametric activation, wrt x.
    """

    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.sign(x.detach()) + (1.0 + a)/2.0


class BentIdentPar(nn.Module):
    """BentIdent parametric activation
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a)/2.0 * (torch.sqrt(x*x + 4.0*self.b*self.b) - 2.0*self.b) + (1.0 + a)/2.0 * x


class BentIdentParDeriv(nn.Module):
    """BentIdent parametric activation deriv
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a)/2.0 * (x / torch.sqrt(x*x + 4.0*self.b*self.b)) + (1.0 + a)/2.0


class DualLeakyReluPar(nn.Module):
    """DualLeakyRelu parametric activation
    """

    def forward(self, x, ap, an):
        return (ap - an) / 2.0 * torch.abs(x) + (ap + an) / 2.0 * x


class DualLeakyReluParDeriv(nn.Module):
    """Derivative of the DualLeakyRelu parametric activation, wrt x.
    """

    def forward(self, x, ap, an):
        return (ap - an) / 2.0 * torch.sign(x.detach()) + (ap + an) / 2.0


class DualBentIdentPar(nn.Module):
    """DualBentIdent parametric activation
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, ap, an):
        return (ap - an) / 2.0 * (torch.sqrt(x * x + 4.0 * self.b * self.b) - 2.0 * self.b) + (ap + an) / 2.0 * x


class DualBentIdentParDeriv(nn.Module):
    """DualBentIdent parametric activation deriv
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, ap, an):
        return (ap - an) / 2.0 * (x / torch.sqrt(x * x + 4.0 * self.b * self.b)) + (ap + an) / 2.0

