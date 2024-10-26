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
import torch.nn.functional as F
from . import filter_layer
from . import activation as activation
from . import fourdim as fourdim
from .plot_corr import plot_global_gocor_weights
from .distance import DistanceMap


class GlobalGOCorOpt(nn.Module):
    """
    Global GOCor optimizer module. 
    Optimizes the GlobalGOCor filter maps on the reference and query images.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        min_filter_reg: an epsilon value to avoid divide by zero
        num_dist_bins: number of bins to compute the distance map
        bin_displacement: displacement bins to compute the distance map

        FOR THE REFERENCE LOSS
        init_gauss_sigma: standard deviation for the initial correlation volume label in the reference image
        v_minus_act: activation function for v_minus weight predictor
        v_minus_init_factor: initialization factor for v_minus weight predictor
        train_label_map: bool to indicate if label map should be train (default is True)

        FOR THE REGULARIZATION LOSS, ie THE QUERY LOSS
        apply_query_loss: bool to apply the query loss
        reg_kernel_size: kernel size of the 4D regularizer operator R_theta
        reg_inter_dim: number of output channels of first 2D convolution of the 4D regularizer operator R_theta
        reg_output_dim: number of output channels of second 2D convolution of the 4D regularizer operator R_theta
    """

    def __init__(self, num_iter=1, init_step_length=1.0, init_filter_reg=1e-2, min_filter_reg=1e-5, steplength_reg=0.0,
                 num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0, v_minus_act='sigmoid',
                 v_minus_init_factor=4.0, train_label_map=True,
                 apply_query_loss=False, reg_kernel_size=3, reg_inter_dim=1, reg_output_dim=1):
        super().__init__()

        self.num_iter = num_iter
        self.min_filter_reg = min_filter_reg
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.steplength_reg = steplength_reg

        # for the query loss L_q
        self.apply_query_loss = apply_query_loss
        if self.apply_query_loss:
            # the 4d conv applied on the cost volume between the filter map and the query features
            self.reg_layer = fourdim.SeparableConv4d(kernel_size=reg_kernel_size, inter_dim=reg_inter_dim,
                                                     output_dim=reg_output_dim,
                                                     bias=False, permute_back_output=False)
            self.reg_layer.weight1.data.normal_(0, 1e-3)
            self.reg_layer.weight2.data.normal_(0, 1e-3)

        # for the reference loss L_r
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        # distance coordinates ==> represents the distance covered by the bins.
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1, -1, 1, 1) * bin_displacement
        # dim is is (1, num_bins, 1, 1) = (output, input, kernel_x, kernel_y), it corresponds to dimension of
        # the weights of the label_map_predictor, spatial_weight_predictor and weight_m_predictor

        # the label_map_predictor, spatial_weight_predictor and weight_m_predictor takes the bin values
        # for a particular pixel as input (corresponding to the distance histogram) and
        # computes the values y, v+ and v- for that location.
        # initialize the label map predictor y'_theta
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
            # the target_map will be an identity matrix. only takes into consideration the first bin. ==> if the pixel
            # is at a distance corresponding to the first bin, the first bin will have value 1, the output will then
            # be 1, otherwise 0.
        else:
            init_gauss = torch.exp(-1 / 2 * (d / init_gauss_sigma) ** 2)
        self.init_gauss = init_gauss  # for plotting
        self.num_bins = num_dist_bins  # for plotting
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        if not train_label_map:
            # option not to train the label map y'
            for param in self.label_map_predictor.parameters():
                param.requires_grad = False

        # initialize the weight v_plus predictor, here called spatial_weight_predictor
        # here, we initialize the weight to a constant of 1. This means that irrespective of the distance (and thus
        # of the histogram/bin distribution), the output will be 1
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # initialize the weights m predictor m_theta, here called target_mask_predictor
        # the weights m at then used to compute the weights v_minus, as v_minus = m * v_plus
        weight_m_predictor_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        # initialization
        self.v_minus_act = v_minus_act
        init_v_minus = v_minus_init_factor * torch.tanh(2.0 - d)
        if v_minus_act == 'sigmoid':
            weight_m_predictor_layers.append(nn.Sigmoid())
            # init weight are passed through sigmoid ==> the final weight m is between 0 and 1
            # initialized so that weight m = 1 at center (match location) and gradually 0 away from match gt.
        elif v_minus_act == 'linear':
            init_v_minus = torch.sigmoid(init_v_minus)
        elif v_minus_act == 'softplus':
            init_v_minus = torch.log(torch.exp( 1/(1 + torch.exp(-init_v_minus))) - 1)
            weight_m_predictor_layers.append(nn.Softplus())
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*weight_m_predictor_layers)
        self.init_target_mask_predictor = init_v_minus.clone()  # for plotting
        self.target_mask_predictor[0].weight.data = init_v_minus

        # initialize activation function sigma (to apply to the correlation score between the filter map and the ref)
        self.score_activation = activation.LeakyReluPar()
        self.score_activation_deriv = activation.LeakyReluParDeriv()

    def _unfold_map(self, full_map):
        output_sz = (full_map.shape[-2] // 2 + 1, full_map.shape[-1] // 2 + 1)
        map_unfold = F.unfold(full_map, output_sz).view(output_sz[0], output_sz[1], output_sz[0], output_sz[1]).flip(
            (2, 3))
        map = map_unfold.permute(2, 3, 0, 1).reshape(1, 1, -1, output_sz[0], output_sz[1])
        return map

    def _plot_weights(self, save_dir):
        plot_global_gocor_weights(save_dir, self.init_gauss, self.label_map_predictor, self.init_target_mask_predictor,
                                  self.target_mask_predictor, self.v_minus_act, self.num_bins,
                                  self.spatial_weight_predictor)

    def forward(self, filter_map, reference_feat, query_feat, num_iter=None, compute_losses=False):
        """
        Apply optimization loop on the initialized filter map
        args:
            filter_map: initial filters, shape is (B, HxW, feat_dim, 1, 1) B=number_of_images*sequences
            reference_feat: features from the reference image,
                            shape is (number_of_images, sequences, feat_dim, H, W), where sequences = b
            query_feat: features from the query image
                        shape is (number_of_images, sequences, feat_dim, H2, W2), where sequences = b
            num_iter: number of iteration, to overwrite num_iter given in init parameters
            compute_losses: compute intermediate losses
        output:
            filters and losses
        """

        if num_iter is None:
            num_iter = self.num_iter

        losses = {'train_reg': [], 'train_reference_loss': [], 'train_query_loss': [], 'train': []}

        num_images = reference_feat.shape[0]
        num_sequences = reference_feat.shape[1] if reference_feat.dim() == 5 else 1
        num_filters = filter_map.shape[1]
        filter_sz = (filter_map.shape[-2], filter_map.shape[-1])  # (fH, fW) filter size
        feat_sz = (reference_feat.shape[-2], reference_feat.shape[-1])  # (H, W)
        output_sz = (feat_sz[0] + (filter_sz[0] + 1) % 2, feat_sz[1] + (filter_sz[1] + 1) % 2)
        # (H, W) because filter size is 1

        assert num_images == 1
        assert num_filters == reference_feat.shape[-2] * reference_feat.shape[-1]
        assert filter_sz[0] % 2 == 1 and filter_sz[1] % 2 == 1  # Assume odd kernels for now

        # Compute distance map
        dist_map_sz = (output_sz[0] * 2 - 1, output_sz[1] * 2 - 1)  # (H*2-1, W*2-1)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(reference_feat.device)
        # at half of the dist_map_sz

        # This first creates a distance map (euclidian distance) from the center point. Then computes the bin values
        # for each of the pixel (corresponding to the distance). For each pixel, the bin values correspond
        # to a histogram of where the distance is (the bin values for each pixel must sum to 1).
        dist_map = self.distance_map(center, dist_map_sz)  # shape is (1, num_bins, dist_map_sz[0], dist_map_sz[1])

        # Compute target map, weights v_plus and weight_m (used in v_minus), used for reference loss
        # first apply the predictor to the dist map, reduces the num_bins dimension to a single value.
        # This corresponds to predicting the v+, m or y based on the distance, because all these functions are
        # parametrized as triangular basis function.
        # ex after self.label_map_predictor(dist_map), shape is (1, 1, dist_map_sz[0], dist_map_sz[1])
        target_map = self._unfold_map(self.label_map_predictor(dist_map))
        # after unfold, shape is (1, 1, H*W, H, W). if gauss is 0, this is basically an identity at initialization

        v_plus = self._unfold_map(self.spatial_weight_predictor(dist_map))
        weight_m = self._unfold_map(self.target_mask_predictor(dist_map))

        # compute regularizer term
        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg ** 2)
        sum_dims = (1, 2) if self.apply_query_loss else 2

        for i in range(num_iter):

            # I. Computing gradient of reference loss with respect to the filter map
            # Computing the cost volume between the filter map and the reference features, dimension (1, b, H*W, H, W)
            scores_filter_w_ref = filter_layer.apply_filter(reference_feat, filter_map)

            # Computing Reference Frame Objective L_R and corresponding gradient with respect to the filter map
            # Applying sigma function on the score:
            act_scores_filter_w_ref = v_plus * self.score_activation(scores_filter_w_ref, weight_m)
            grad_act_scores_by_filter = v_plus * self.score_activation_deriv(scores_filter_w_ref, weight_m)
            loss_ref_residuals = act_scores_filter_w_ref - v_plus * target_map
            mapped_residuals = grad_act_scores_by_filter * loss_ref_residuals

            # Computing the gradient of the reference loss with respect to the filer map
            filter_grad_loss_ref = filter_layer.apply_feat_transpose(reference_feat, mapped_residuals, filter_sz,
                                                                     training=self.training)

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
                # Computing the cost volume between the filter map and the query features, dimension (1, b, H*W, H2, W2)
                scores_filter_w_query = filter_layer.apply_filter(query_feat, filter_map)

                # Applying the 4D kernel on the cost volume,
                # output shape is (b, H2, W2, output_dim, H, W) because self.reg_layer.permute_back_output = False
                loss_query_residuals = self.reg_layer(scores_filter_w_query.reshape(-1, *feat_sz, *feat_sz))

                # Computing the gradient of the query loss with respect to the filer map
                # apply transpose 4D convolution, returns to (1, b, H*W, H2, W2)
                reg_tp_res = self.reg_layer(loss_query_residuals, transpose=True).reshape(scores_filter_w_query.shape)

                filter_grad_loss_query = filter_layer.apply_feat_transpose(query_feat, reg_tp_res, filter_sz,
                                                                           training=self.training)
                filter_grad += filter_grad_loss_query
                if compute_losses:
                    # compute corresponding loss
                    loss_query = 0.5 * (loss_query_residuals ** 2).sum() / num_sequences

            if compute_losses:
                losses['train_reference_loss'].append(loss_ref)
                losses['train_reg'].append(loss_reg)
                losses['train_query_loss'].append(loss_query)
                losses['train'].append(losses['train_reference_loss'][-1] + losses['train_reg'][-1] +
                                       losses['train_query_loss'][-1])

            # III. Calculating alpha denominator
            # 1. Reference loss (L_r)
            # Computing the cost volume between the gradient of the loss with respect to the filter map with
            # the reference features in scores_filter_grad_w_ref
            scores_filter_grad_w_ref = filter_layer.apply_filter(reference_feat, filter_grad)
            alpha_den_loss_ref_residuals = grad_act_scores_by_filter * scores_filter_grad_w_ref
            alpha_den = (alpha_den_loss_ref_residuals * alpha_den_loss_ref_residuals)\
                .view(num_sequences, num_filters, -1).sum(dim=sum_dims)

            # 2. Query Loss (L_q)
            if self.apply_query_loss:
                # Hessian parts for regularization
                scores_filter_grad_w_query = filter_layer.apply_filter(query_feat, filter_grad)
                alpha_den_loss_query_residual = self.reg_layer(scores_filter_grad_w_query
                                                               .reshape(-1, *feat_sz, *feat_sz))

                alpha_den += (alpha_den_loss_query_residual * alpha_den_loss_query_residual)\
                    .view(num_sequences, num_filters, -1).sum(dim=sum_dims)

            # IV. Compute step length alpha
            alpha_num = (filter_grad * filter_grad).reshape(num_sequences, num_filters, -1).sum(dim=sum_dims)
            alpha_den = (alpha_den + reg_weight * alpha_num).clamp(1e-8)

            if self.steplength_reg > 0:
                alpha_den = alpha_den + self.steplength_reg * alpha_num
            alpha = alpha_num / alpha_den

            # V. Update filter map, filter map shape is b, H*W, feat_dim, 1, 1
            if self.apply_query_loss:
                # alpha is b
                filter_map = filter_map - (step_length * alpha.view(num_sequences, 1, 1, 1, 1)) * filter_grad
            else:
                # alpha is b, H*W
                filter_map = filter_map - (step_length * alpha.view(num_sequences, num_filters, 1, 1, 1)) * filter_grad

        if compute_losses:
            print('GlobalGOCor: train reference loss is {}'.format(losses['train_reference_loss']))
            print('GlobalGOCor: train query loss is {}'.format(losses['train_query_loss']))
            print('GlobalGOCor: train reg is {}\n'.format(losses['train_reg']))

        return filter_map, losses


class GlobalGOCor(nn.Module):
    """The main GlobalGOCor module for computing the correlation volume, as a replacement to the feature correlation
    layer.
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
        put_query_feat_in_channel_dimension: set order of the output. The feature dimension consists of the ref image
                                             coordinates if False and the query image coordinates if True.
                                             (default: True)
    """
    def __init__(self, filter_initializer, filter_optimizer,
                 put_query_feat_in_channel_dimension=True, post_processing=None):
        super(GlobalGOCor, self).__init__()

        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.put_query_feat_in_channel_dimension = put_query_feat_in_channel_dimension
        self.post_processing = post_processing

    def forward(self, reference_feat, query_feat, training=True, **kwargs):
        """
        Computes the GOCor correspondence volume between inputted reference and query feature maps.
        args:
            reference_feat: reference feature with shape (b, feat_dim, H, W)
            query_feat: query feature with shape (b, feat_dim, H2, W2)
            training: True

        output:
            scores: correspondence volume between the optimized filter map (instead of the reference features in the
                    feature correlation layer) and the query feature map.
                    shape is (b, H2*W2, H, W) if self.put_query_feat_in_channel_dimension is True,
                    else shape is (b, H*W, H2, W2)
            losses: dictionary containing the losses computed during optimization
        """

        # reshape both feature maps to size (number_of_images, sequences, feat_dim, H,W), where sequences = b
        reference_feat = reference_feat.view(1, *reference_feat.shape[-4:])
        query_feat = query_feat.view(1, *query_feat.shape[-4:])

        # initialize filter map, resulting filter has shape (B, HxW, feat_dim, 1, 1) B=number_of_images*sequences
        # this is because one wants a filter vector for each position of the reference feature map (HxW positions).
        filter = self.filter_initializer(reference_feat)

        # optimizes the filter map
        filter, losses = self.filter_optimizer(filter, reference_feat, query_feat=query_feat, **kwargs)

        # with the resulting optimized filter map:
        # computes the correspondence volume between the filter map and the query features
        scores = filter_layer.apply_filter(query_feat, filter)
        # resulting shape (1, B, HxW, H2, W2), but here filter_map is in channel dimension !
        scores = torch.squeeze(scores, 0)  # shape is (b, H*W, H2, W2)

        if self.put_query_feat_in_channel_dimension:
            # put query feat (query image) in channel dimension, resulting shape is (B, H2xW2, H, W)
            scores = scores.view(scores.shape[0], *reference_feat.shape[-2:], -1).permute(0, 3, 1, 2).contiguous()

        if self.post_processing == 'add_corr':
            # compute also for the other direction
            filter_source_image = self.filter_initializer(query_feat)
            filter_source_image, losses_source_image = self.filter_optimizer(filter_source_image, query_feat, test_feat=reference_feat, **kwargs)
            scores_source_image_transpose = filter_layer.apply_filter(reference_feat, filter_source_image)
            scores_source_image_transpose = torch.squeeze(scores_source_image_transpose, 0)  # shape is (b, H*W, H, W)
            # here source_image is in channel dimension already, no need to exchange dimension
            scores = scores + scores_source_image_transpose
        elif self.post_processing == 'leaky_relu_add_corr':
            filter_source_image = self.filter_initializer(query_feat)
            filter_source_image, losses_source_image = self.filter_optimizer(filter_source_image, query_feat, test_feat=reference_feat, **kwargs)
            scores_source_image_transpose = filter_layer.apply_filter(reference_feat, filter_source_image)
            scores_source_image_transpose = torch.squeeze(scores_source_image_transpose, 0)  # shape is (b, H*W, H, W)
            # here source_image is in channel dimension already, no need to exchange dimension
            scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.1) + \
                     torch.nn.functional.leaky_relu(scores_source_image_transpose, negative_slope=0.1)

        return scores, losses


######## Example ########
# see global_gocor_modules.py
