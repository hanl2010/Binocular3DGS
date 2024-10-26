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

import math
import torch
import torch.nn as nn
from . import activation as activation
from .distance import DistanceMap
from .local_correlation.correlation import FunctionCorrelation, FunctionCorrelationTranspose
from .plot_corr import plot_local_gocor_weights
from . import fourdim as fourdim


class LocalCorrInitializerZeros(nn.Module):
    """Local GOCor initializer module. 
    Initializes the Local GOCor filter with a zero tensor.
    args:
        filter_size: spatial kernel size of filter
    """

    def __init__(self, filter_size=1):
        super().__init__()
        assert filter_size == 1

        self.filter_size = filter_size

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, feat_dim, H, W)
        """
        weights = torch.zeros_like(feat)
        return weights


class LocalCorrSimpleInitializer(nn.Module):
    """Local GOCor initializer module. 
    Initializes the Local GOCor filter through a simple norm operation
    args:
        filter_size: spatial kernel size of filter
    """

    def __init__(self, filter_size=1):
        super().__init__()
        assert filter_size == 1

        self.filter_size = filter_size
        self.scaling = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, feat_dim, H, W)
        """

        weights = feat / ((feat*feat).mean(dim=1, keepdim=True) + 1e-6)
        weights = self.scaling * weights
        return weights


class LocalCorrContextAwareInitializer(nn.Module):
    """Local GOCor initializer module. 
    Initializes the Local GOCor filter ContextAwareInitializer.
    It assumes that the filter at a particular pixel location, correlated with the features at the same location
    should be equal to 1 (here the value 1 islearnt as target_fg_value), while correlated with features
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
        self.target_fg = nn.Parameter(init_fg * torch.ones(1, float))
        self.target_bg = nn.Parameter(init_bg * torch.ones(1, float))

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, feat_dim, H, W)
        """

        d = feat.size(1)
        bg_weights = feat.mean(dim=2, keepdim=True) # averages over all features

        ff = (feat * feat).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (feat * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg * bb - self.target_bg * fb
        bg_scale = self.target_fg * fb - self.target_bg * ff
        weights = d * (fg_scale * feat - bg_scale * bg_weights) / (den + 1e-6)
        return weights


class LocalCorrFlexibleContextAwareInitializer(nn.Module):
    """Local GOCor initializer module. 
    Initializes the Local GOCor with a Flexible-ContextAwareInitializer.
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

    def forward(self, feat):
        """Initialize filter.
        args:
            feat: input features (sequences, feat_dim, H, W)
        output:
            weights: initial filters (sequences, feat_dim, H, W)
        """

        d = feat.size(1)
        bg_weights = feat.mean(dim=2, keepdim=True)  # averages over all features

        ff = (feat * feat).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (feat * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg.view(d, 1, 1) * bb - self.target_bg.view(d, 1, 1) * fb
        bg_scale = self.target_fg.view(d, 1, 1) * fb - self.target_bg.view(d, 1, 1) * ff
        weights = d * (fg_scale * feat - bg_scale * bg_weights) / (den + 1e-6)
        return weights


class LocalGOCorrOpt(nn.Module):
    """Local GOCor optimizer module. 
    Optimizes the LocalGOCor filter map on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=3, init_step_length=1.0, init_filter_reg=1e-2,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0,
                 v_minus_act='sigmoid', v_minus_init_factor=4.0, search_size=9,
                 apply_query_loss=False, reg_kernel_size=3, reg_inter_dim=1, reg_output_dim=1):
        super().__init__()

        assert search_size == 9  # fixed to 9 currently, we are working on making a general version

        self.num_iter = num_iter
        self.min_filter_reg = min_filter_reg
        self.search_size = search_size

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        
        # for the query loss L_q
        # not used in final version, because too computationally expensive
        self.apply_query_loss = apply_query_loss
        if self.apply_query_loss:
            # the 4d conv applied on the correlation filter with query
            self.reg_layer = fourdim.SeparableConv4d(kernel_size=reg_kernel_size, inter_dim=reg_inter_dim,
                                                     output_dim=reg_output_dim,
                                                     bias=False, permute_back_output=False)
            self.reg_layer.weight1.data.normal_(0, 1e-3)
            self.reg_layer.weight2.data.normal_(0, 1e-3)
        
        # for the reference loss L_r
        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        
        # initialize the label map predictor y'_theta
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)
        self.init_gauss = init_gauss
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        
        # initialize the weight v_plus predictor, here called spatial_weight_predictor
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        
        # initialize the weights m predictor m_theta, here called target_mask_predictor
        # the weights m at then used to compute the weights v_minus, as v_minus = m * v_plus
        self.num_bins = num_dist_bins
        init_v_minus = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        init_w = v_minus_init_factor * torch.tanh(2.0 - d)
        self.v_minus_act = v_minus_act
        if v_minus_act == 'sigmoid':
            init_v_minus.append(nn.Sigmoid())
        elif v_minus_act == 'linear':
            init_w = torch.sigmoid(init_w)
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*init_v_minus)
        self.target_mask_predictor[0].weight.data = init_w
        self.init_target_mask_predictor = init_w.clone()  # for plotting
        
        # initialize activation function sigma (to apply to the correlation score between the filter map and the ref)
        self.score_activation = activation.LeakyReluPar()
        self.score_activation_deriv = activation.LeakyReluParDeriv()

    def _plot_weights(self, save_dir):
        plot_local_gocor_weights(save_dir, self.init_gauss, self.label_map_predictor, self.init_target_mask_predictor,
                                 self.target_mask_predictor, self.v_minus_act, self.num_bins,
                                 self.spatial_weight_predictor)

    def forward(self, filter_map, reference_feat, query_feat=None, num_iter=None, compute_losses=False):
        """
        Apply optimization loop on the initialized filter map
        args:
            filter_map: initial filters, shape is (b, feat_dim, H, W)
            reference_feat: features from the reference image, shape is (b, feat_dim, H, W)
            query_feat: features from the query image, shape is (b, feat_dim, H, W)
            num_iter: number of iteration, to overwrite num_iter given in init parameters
            compute_losses: compute intermediate losses
        output:
            filters and losses
        """
        if num_iter is None:
            num_iter = self.num_iter

        num_sequences = reference_feat.shape[0]
        num_filters = reference_feat.shape[-2] * reference_feat.shape[-1]
        feat_sz = (reference_feat.shape[-2], reference_feat.shape[-1])
        feat_dim = reference_feat.shape[-3]

        # Compute distance map
        dist_map_sz = (self.search_size, self.search_size)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(reference_feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute target map, weights v_plus and weight_m (used in v_minus), used for reference loss
        target_map = self.label_map_predictor(dist_map).reshape(1, -1, 1, 1)
        v_plus = self.spatial_weight_predictor(dist_map).reshape(1, -1, 1, 1)
        weight_m = self.target_mask_predictor(dist_map).reshape(1, -1, 1, 1)

        # compute regularizer term
        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)/(feat_dim**2)

        losses = {'train': [], 'train_reference_loss': [], 'train_reg': [], 'train_query_loss': []}

        for i in range(num_iter):
            # I. Computing gradient of reference loss with respect to the filter map
            # Computing the cost volume between the filter map and the reference features
            scores_filter_w_ref = FunctionCorrelation(filter_map, reference_feat)

            # Computing Reference Frame Objective L_R and corresponding gradient with respect to the filter map
            # Applying sigma function on the score:
            act_scores_filter_w_ref = v_plus * self.score_activation(scores_filter_w_ref, weight_m)
            grad_act_scores_by_filter = v_plus * self.score_activation_deriv(scores_filter_w_ref, weight_m)
            loss_ref_residuals = act_scores_filter_w_ref - v_plus * target_map
            mapped_residuals = grad_act_scores_by_filter * loss_ref_residuals

            # Computing the gradient of the reference loss with respect to the filer map
            filter_grad_loss_ref = FunctionCorrelationTranspose(mapped_residuals, reference_feat)

            # Computing the gradient of the regularization term with respect to the filter map
            filter_grad_reg = reg_weight * filter_map

            filter_grad = filter_grad_reg + filter_grad_loss_ref

            if compute_losses:
                # compute corresponding loss
                loss_ref = 0.5 * (loss_ref_residuals**2).sum()/num_sequences
                loss_reg = 0.5 / reg_weight.item() * (filter_grad_reg ** 2).sum() / num_sequences

            # II. Computing Query Frame Objective L_q and corresponding gradient with respect to the filter map
            loss_query = 0
            if self.apply_query_loss:
                # Computing the cost volume between the filter map and the query features
                # dimension (b, search_size*search_size, H, W)
                scores_filter_w_query = FunctionCorrelation(filter_map, query_feat)

                # Applying the 4D kernel on the cost volume,
                loss_query_residuals = self.reg_layer(scores_filter_w_query.reshape(-1, self.search_size,
                                                                                    self.search_size, *feat_sz))
                # output shape is (b, H, W, output_dim, search_size, search_size)

                #  Computing the gradient of the query loss with respect to the filer map
                # apply transpose convolution, returns to b, search_size, search_size, H, W
                reg_tp_res = self.reg_layer(loss_query_residuals, transpose=True).reshape(scores_filter_w_query.shape)

                filter_grad_loss_query = FunctionCorrelationTranspose(reg_tp_res, query_feat)
                filter_grad += filter_grad_loss_query
                if compute_losses:
                    # calculate the corresponding loss:
                    loss_query = 0.5 * (loss_query_residuals ** 2).sum() / num_sequences

            # III. Calculating alpha denominator
            # 1. Reference loss (L_r)
            # Computing the cost volume between the gradient of the loss with respect to the filter map with
            # the reference features in scores_filter_grad_w_ref
            scores_filter_grad_w_ref = FunctionCorrelation(filter_grad, reference_feat)
            scores_filter_grad_w_ref = grad_act_scores_by_filter * scores_filter_grad_w_ref
            if self.apply_query_loss:
                alpha_den = (scores_filter_grad_w_ref * scores_filter_grad_w_ref).view(num_sequences, -1).sum(dim=1)
                # shape is b
            else:
                alpha_den = (scores_filter_grad_w_ref * scores_filter_grad_w_ref).sum(dim=1, keepdim=True)
                # shape is b, spa**2, H, W

            # 2. Query Loss (L_q)
            if self.apply_query_loss:
                # Hessian parts for regularization
                scores_filter_grad_w_query = FunctionCorrelation(filter_grad, query_feat)
                alpha_den_loss_query_residual = self.reg_layer(scores_filter_grad_w_query.reshape(-1,
                                                                                                  self.search_size,
                                                                                                  self.search_size,
                                                                                                  *feat_sz))
                alpha_den += (alpha_den_loss_query_residual * alpha_den_loss_query_residual)\
                    .view(num_sequences, -1).sum(dim=1)

            # IV. Compute step length alpha
            if self.apply_query_loss:
                alpha_num = (filter_grad * filter_grad).view(num_sequences, -1).sum(dim=1)
            else:
                alpha_num = (filter_grad * filter_grad).sum(dim=1, keepdim=True)
            alpha_den = (alpha_den + reg_weight * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # V. Update filter map
            if self.apply_query_loss:
                filter_map = filter_map - (step_length * alpha.view(num_sequences, 1, 1, 1)) * filter_grad
            else:
                filter_map = filter_map - (step_length * alpha) * filter_grad

            if compute_losses:
                losses['train_reference_loss'].append(loss_ref)
                losses['train_reg'].append(loss_reg)
                losses['train_query_loss'].append(loss_query)
                losses['train'].append(losses['train_reference_loss'][-1] + losses['train_reg'][-1] +
                                       losses['train_query_loss'][-1])

        if compute_losses:
            print('LocalGOCor: train reference loss is {}'.format(losses['train_reference_loss']))
            print('LocalGOCor: train query loss is {}'.format(losses['train_query_loss']))
            print('LocalGOCor: train reg is {}\n'.format(losses['train_reg']))

        return filter_map, losses


class LocalGOCor(nn.Module):
    """The main LocalGOCor module for computing the local correlation volume.
    For now, only supports local search radius of 4. 
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
    """
    def __init__(self, filter_initializer, filter_optimizer):
        super(LocalGOCor, self).__init__()

        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer

    def forward(self, reference_feat, query_feat, **kwargs):
        """
        Computes the local GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)

        output:
            scores: local correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
        """
        
        # initializes the filter map
        filter = self.filter_initializer(reference_feat)
        
        # optimizes the filter map
        filter, losses = self.filter_optimizer(filter, reference_feat, query_feat=query_feat, **kwargs)
        
        # compute the local cost volume between optimized filter map and query features
        scores = FunctionCorrelation(filter, query_feat)

        return scores



######## Example ########
#
# initializer = LocalCorrSimpleInitializer()
#
# optimizer = LocalGOCorrOpt(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
#                            num_dist_bins=num_dist_bins, bin_displacement=bin_displacement,
#                            v_minus_act=v_minus_act, v_minus_init_factor=v_minus_init_factor, search_size=search_size,
#                            apply_query_loss=False, reg_kernel_size=1, reg_inter_dim=1, reg_output_dim=1)

# corr_module = LocalGOCor(filter_initializer=initializer, filter_optimizer=optimizer)
