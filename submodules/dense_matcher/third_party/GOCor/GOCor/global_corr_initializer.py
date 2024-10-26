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


class GlobalCorrSimpleInitializer(nn.Module):
    """Global GOCor initializer module.
    Initializes the GlobalGOCor filter through simple norm operation.
    It corresponds to ideal case where scalar product between filter at a particular location and
    features at other locations is equal to 0, they are orthogonal.
    args:
        filter_size: spatial kernel size of filter (fH, fW)
    """

    def __init__(self, filter_size=1):
        super().__init__()

        self.filter_size = filter_size
        self.scaling = nn.Parameter(torch.ones(1))
        # only learns the value of scaling !!!!

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, num_filters, feat_dim, fH, fW) where num_filters=H*W
        """

        feat = feat.view(-1, *feat.shape[-3:])
        weights = F.unfold(feat, self.filter_size, padding=self.filter_size//2)

        weights = weights / (weights*weights).sum(dim=1,keepdim=True)
        # weights here are features/norm(feature)**2

        weights = self.scaling * weights.permute(0,2,1).reshape(feat.shape[0],
                                                                feat.shape[-2]*feat.shape[-1],
                                                                feat.shape[-3],
                                                                self.filter_size,
                                                                self.filter_size).contiguous()

        return weights


class GlobalCorrFlexibleSimpleInitializer(nn.Module):
    """Global GOCor initializer module.
    Initializes the GlobalGOCor filter through a norm operation with more learnable parameters.
    It corresponds to ideal case where scalar product between filter at a particular location and
    features at other locations is equal to 0, they are orthogonal.
    args:
        filter_size: spatial kernel size of filter
        number_feat: dimensionality of input features
    """

    def __init__(self, filter_size=1, number_feat=512):
        super().__init__()

        self.filter_size = filter_size
        self.scaling = nn.Parameter(1.0 * torch.ones(number_feat))
        # learns the whole vector, corresponding to the number of features

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, num_filters, feat_dim, fH, fW) where num_filters=H*W
        """

        feat = feat.view(-1, *feat.shape[-3:])
        d = feat.size(1)
        weights = F.unfold(feat, self.filter_size, padding=self.filter_size//2)

        weights = weights / (weights*weights).sum(dim=1,keepdim=True)
        # weights here are features/norm(feature)**2

        weights = self.scaling.view(d, self.filter_size, self.filter_size) * \
                  weights.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-2]*feat.shape[-1],
                                                   feat.shape[-3], self.filter_size, self.filter_size).contiguous()

        return weights


class GlobalCorrContextAwareInitializer(nn.Module):
    """Global GOCor initializer module.
    Initializes the GlobalGOCor filter with ContextAwareInitializer.
    It assumes that the filter at a particular pixel location, correlated with the features at the same location
    should be equal to 1 (here the value 1 is learnt as target_fg_value), while correlated with features
    at other locations should be zero (here the value 0 is learnt as target_bg). The other features locations are
    approximated by the mean of the features, called background_vector.
    Filter at particular location should be linear combination of feature at this location (foreground) and
    background features (average of all features)

    It corresponds to non ideal cases, where scalar product between filter and background feature is not
    necessarily equal to 0.
    args:
        filter_size: spatial kernel size of filter
        init_fg: initial value for scalar product between filter and features at the same location (=1)
        init_bg: initial value for scalar product between filter and background features (=0)
    """

    def __init__(self, filter_size=1, init_fg=1.0, init_bg=0.0):
        super().__init__()

        self.filter_size = filter_size
        self.target_fg = nn.Parameter(torch.Tensor([init_fg]))
        self.target_bg = nn.Parameter(torch.Tensor([init_bg]))
        # learns the value of those two parameters !!

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, num_filters, feat_dim, fH, fW) where num_filters=H*W
        """

        feat = feat.view(-1, *feat.shape[-3:])
        weights = F.unfold(feat, self.filter_size, padding=self.filter_size // 2)

        bg_weights = weights.mean(dim=2, keepdim=True)  # averages over all features

        ff = (weights * weights).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (weights * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg * bb - self.target_bg * fb
        bg_scale = self.target_fg * fb - self.target_bg * ff
        weights = (fg_scale * weights - bg_scale * bg_weights) / den

        weights = weights.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-2] * feat.shape[-1], feat.shape[-3],
                                                   self.filter_size, self.filter_size).contiguous()
        return weights


class GlobalCorrFlexibleContextAwareInitializer(nn.Module):
    """Global GOCor initializer module.
    Initializes the GlobalGOCor filter with Flexible-ContextAwareInitializer.
    It assumes that the filter at a particular pixel location, correlated with the features at the same location
    should be equal to 1 (here the value 1 is a vector, learnt as target_fg_value), while correlated with features
    at other locations should be zero (here the value 0 is a vector, learnt as target_bg). The other features locations are
    approximated by the mean of the features, called background_vector.
    Filter at particular location should be linear combination of feature at this location (foreground) and
    background features (average of all features)

    It corresponds to non ideal cases, where scalar product between filter and background feature is not
    necessarily equal to 0.
    args:
        filter_size: spatial kernel size of filter
        number_feat: dimensionality of input features
        init_fg: initial value for scalar product between filter and features at the same location (=1)
        init_bg: initial value for scalar product between filter and background features (=0)
    """

    def __init__(self, filter_size=1, number_feat=512, init_fg=1.0, init_bg=0.0):
        super().__init__()

        self.filter_size = filter_size
        self.target_fg = nn.Parameter(init_fg * torch.ones(number_feat))
        self.target_bg = nn.Parameter(init_bg * torch.ones(number_feat))
        # these two values are vectors, instead of scalars as in ContextAwareInitializer

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W), feat_dim = d
        output:
            weights: initial filters (sequences, num_filters, feat_dim, fH, fW) where num_filters=H*W
        """

        feat = feat.view(-1, *feat.shape[-3:])
        # shape is sequences x d x H x W
        d = feat.size(1)

        weights = F.unfold(feat, self.filter_size, padding=self.filter_size // 2)
        # weights shape is sequences x d x (HxW)

        bg_weights = weights.mean(dim=2, keepdim=True) # averages over all features

        ff = (weights * weights).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (weights * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg.view(d, self.filter_size) * bb - self.target_bg.view(d, self.filter_size) * fb
        bg_scale = self.target_fg.view(d, self.filter_size) * fb - self.target_bg.view(d, self.filter_size) * ff
        weights = (fg_scale * weights - bg_scale * bg_weights) / den

        weights = weights.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-2] * feat.shape[-1], feat.shape[-3],
                                                   self.filter_size, self.filter_size).contiguous()
        return weights


class GlobalCorrInitializerZero(nn.Module):
    """Global GOCor initializer module.
    Initializes the GlobalGOCor filter with a zero tensor
    args:
        filter_size: spatial kernel size of filter
    """
    def __init__(self, filter_size=1):
        super().__init__()

        self.filter_size = filter_size

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, num_filters, feat_dim, fH, fW) where num_filters=H*W
        """

        weights = torch.zeros(feat.shape[0], feat.shape[-2] * feat.shape[-1], feat.shape[-3],
                              self.filter_size, self.filter_size).cuda()
        # weights is sequences, HxW, feat_dim, 1, 1
        return weights
