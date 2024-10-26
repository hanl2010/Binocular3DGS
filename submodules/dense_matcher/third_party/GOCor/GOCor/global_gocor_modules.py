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

from . import global_gocor
from . import global_corr_initializer
import torch.nn as nn
from .optimizer_selection_functions import define_optimizer_global_corr


class GlobalGOCorWithSimpleInitializer(nn.Module):
    """The main GlobalGOCor module for computing the correlation volume, with filter initializer module the
    GlobalGOCorSimpleInitializer
    args:
        global_gocor_arguments: dictionary containing arguments for the gocor module
        filter_size: for the filter map, only supports 1 for now
        put_query_feat_in_channel_dimension: set order of the output. The feature dimension consists of the ref image
                                             coordinates if False and the query image coordinates if True.
                                             (default: True)
    """

    def __init__(self, global_gocor_arguments, filter_size=1,
                 put_query_feat_in_channel_dimension=True):
        super(GlobalGOCorWithSimpleInitializer, self).__init__()

        initializer = global_corr_initializer.GlobalCorrSimpleInitializer(filter_size=filter_size)

        optimizer = define_optimizer_global_corr(global_gocor_arguments)

        corr_module = global_gocor.GlobalGOCor(filter_initializer=initializer, filter_optimizer=optimizer,
                                               put_query_feat_in_channel_dimension=put_query_feat_in_channel_dimension)
        self.corr_module = corr_module

    def forward(self, reference_feat, query_feat, **kwargs):
        """
        Computes the GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
                    shape is (b, H2*W2, H, W) if self.put_query_feat_in_channel_dimension is True,
                    else shape is (b, H*W, H2, W2)
            losses: dictionary containing the losses computed during optimization
        """

        return self.corr_module(reference_feat, query_feat, **kwargs)


class GlobalGOCorWithFlexibleSimpleInitializer(nn.Module):
    """The main GlobalGOCor module for computing the correlation volume, with filter initializer module the
    GlobalGOCorFlexibleSimpleInitializer
    args:
        global_gocor_arguments: dictionary containing arguments for the gocor module
        filter_size: for the filter map, only supports 1 for now
        put_query_feat_in_channel_dimension: set order of the output. The feature dimension consists of the ref image
                                             coordinates if False and the query image coordinates if True.
                                             (default: True)
    """

    def __init__(self, global_gocor_arguments, filter_size=1, put_query_feat_in_channel_dimension=True):
        super(GlobalGOCorWithFlexibleSimpleInitializer, self).__init__()

        num_features = global_gocor_arguments['num_features'] if 'num_features' \
                                                                 in list(global_gocor_arguments.keys()) else 512
        initializer = global_corr_initializer.GlobalCorrFlexibleSimpleInitializer(filter_size=filter_size,
                                                                                  number_feat=num_features)

        optimizer = define_optimizer_global_corr(global_gocor_arguments)

        corr_module = global_gocor.GlobalGOCor(filter_initializer=initializer, filter_optimizer=optimizer,
                                               put_query_feat_in_channel_dimension=put_query_feat_in_channel_dimension)
        self.corr_module = corr_module

    def forward(self, reference_feat, query_feat, **kwargs):
        """
        Computes the GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
                    shape is (b, H2*W2, H, W) if self.put_query_feat_in_channel_dimension is True,
                    else shape is (b, H*W, H2, W2)
            losses: dictionary containing the losses computed during optimization
        """

        return self.corr_module(reference_feat, query_feat, **kwargs)


class GlobalGOCorWithContextAwareInitializer(nn.Module):
    """The main GlobalGOCor module for computing the correlation volume, with filter initializer module the
    GlobalGOCorContextAwareInitializer
    args:
        global_gocor_arguments: dictionary containing arguments for the gocor module
        filter_size: for the filter map, only supports 1 for now
        put_query_feat_in_channel_dimension: set order of the output. The feature dimension consists of the ref image
                                             coordinates if False and the query image coordinates if True.
                                             (default: True)
    """
    def __init__(self, global_gocor_arguments, filter_size=1, put_query_feat_in_channel_dimension=True):
        super(GlobalGOCorWithContextAwareInitializer, self).__init__()

        initializer = global_corr_initializer.GlobalCorrContextAwareInitializer(filter_size=filter_size)

        optimizer = define_optimizer_global_corr(global_gocor_arguments)

        corr_module = global_gocor.GlobalGOCor(filter_initializer=initializer, filter_optimizer=optimizer,
                                               put_query_feat_in_channel_dimension=put_query_feat_in_channel_dimension)
        self.corr_module = corr_module

    def forward(self, reference_feat, query_feat, **kwargs):
        """
        Computes the GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
                    shape is (b, H2*W2, H, W) if self.put_query_feat_in_channel_dimension is True,
                    else shape is (b, H*W, H2, W2)
            losses: dictionary containing the losses computed during optimization
        """

        return self.corr_module(reference_feat, query_feat, **kwargs)


class GlobalGOCorWithFlexibleContextAwareInitializer(nn.Module):
    """The main GlobalGOCor module for computing the correlation volume, with filter initializer module the
    GlobalGOCorFlexibleContextAwareInitializer
    args:
        global_gocor_arguments: dictionary containing arguments for the gocor module
        filter_size: for the filter map, only supports 1 for now
        out_feature_dim: feature dimension
        put_query_feat_in_channel_dimension: set order of the output. The feature dimension consists of the ref image
                                             coordinates if False and the query image coordinates if True.
                                             (default: True)
    """
    def __init__(self, global_gocor_arguments, filter_size=1, put_query_feat_in_channel_dimension=True):
        super(GlobalGOCorWithFlexibleContextAwareInitializer, self).__init__()

        num_features = global_gocor_arguments['num_features'] if 'num_features' \
                                                                 in list(global_gocor_arguments.keys()) else 512
        initializer = global_corr_initializer.GlobalCorrFlexibleContextAwareInitializer(number_feat=num_features,
                                                                                        filter_size=filter_size)

        optimizer = define_optimizer_global_corr(global_gocor_arguments)

        # apply the regularizer to the test score also
        if 'corr_post_processing' in list(global_gocor_arguments.keys()):
            post_processing = global_gocor_arguments['corr_post_processing']
        else:
            post_processing = None
        corr_module = global_gocor.GlobalGOCor(filter_initializer=initializer, filter_optimizer=optimizer,
                                               put_query_feat_in_channel_dimension=put_query_feat_in_channel_dimension,
                                               post_processing=post_processing)
        self.corr_module = corr_module

    def forward(self, reference_feat, query_feat, **kwargs):
        """
        Computes the GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
                    shape is (b, H2*W2, H, W) if self.put_query_feat_in_channel_dimension is True,
                    else shape is (b, H*W, H2, W2)
            losses: dictionary containing the losses computed during optimization
        """

        return self.corr_module(reference_feat, query_feat, **kwargs)

