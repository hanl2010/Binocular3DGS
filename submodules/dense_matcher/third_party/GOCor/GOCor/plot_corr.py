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
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import unravel_index


def plot_global_gocor_weights(save_dir, init_gauss, target_map_predictor, init_weight_m_predictor,
                              weight_m_predictor, v_minus_act, num_bins, v_plus_predictor):
    # target_mask_predictor, label_map_predictor and spatial_weights_predictors are shape 1 x num_dist_bin 1 x 1
    x = np.arange(0, 10)
    fig, axis = plt.subplots(3, 1, figsize=(20, 20))
    target_predictor_initial = (init_gauss - init_gauss.min()).squeeze(0).squeeze(-1).cpu().numpy()
    target_predictor = target_map_predictor.weight.data.clone().detach().squeeze(0).squeeze(-1).cpu().numpy()
    axis[0].plot(x, target_predictor_initial, color='green')
    axis[0].plot(x, target_predictor, color='red')
    axis[0].set_title('target map predictor weights for y\', \ninitial weights in green min = {:0.2f}, max={:0.2f}'
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(target_predictor_initial), 2),
        np.round(np.amax(target_predictor_initial), 2),
        np.round(np.amin(target_predictor), 2),
        np.round(np.amax(target_predictor), 2)))

    weight_m_predictor_initial = init_weight_m_predictor
    weight_m_predictor = weight_m_predictor[0].weight.data.clone().detach()
    if v_minus_act == 'sigmoid':
        weight_m_predictor_initial = torch.sigmoid(weight_m_predictor_initial)
        weight_m_predictor = torch.sigmoid(weight_m_predictor)
    if v_minus_act == 'softplus':
        weight_m_predictor_initial = torch.nn.functional.softplus(weight_m_predictor_initial)
        weight_m_predictor = torch.nn.functional.softplus(weight_m_predictor)
    weight_m_predictor_initial = weight_m_predictor_initial.squeeze(0).squeeze(-1).cpu().numpy()
    weight_m_predictor = weight_m_predictor.squeeze(0).squeeze(-1).cpu().numpy()

    axis[1].plot(x, weight_m_predictor_initial, color='green')
    axis[1].plot(x, weight_m_predictor, color='red')
    axis[1].set_title('m predictor weights used for v_minus, \ninitial weights in green min = {:0.2f}, max={:0.2f} '
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(weight_m_predictor_initial), 2),
        np.round(np.amax(weight_m_predictor_initial), 2),
        np.round(np.amin(weight_m_predictor), 2),
        np.round(np.amax(weight_m_predictor), 2)))

    v_plus_predictor_initial = np.ones(num_bins)
    v_plus_predictor = v_plus_predictor.weight.data.clone().detach().squeeze(0).squeeze(
        -1).cpu().numpy()
    axis[2].plot(x, v_plus_predictor_initial, color='green')
    axis[2].plot(x, v_plus_predictor, color='red')
    axis[2].set_title('v_plus predictor weights, \ninitial weights in green min = {:0.2f}, max={:0.2f} '
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(v_plus_predictor_initial), 2),
        np.round(np.amax(v_plus_predictor_initial), 2),
        np.round(np.amin(v_plus_predictor), 2),
        np.round(np.amax(v_plus_predictor), 2)))

    fig.savefig('{}/{}.png'.format(save_dir, 'GlobalGOCor_weights_predictors'), bbox_inches='tight')
    plt.close(fig)
    print('plotted_weights')


def plot_local_gocor_weights(save_dir, init_gauss, target_map_predictor, init_weight_m_predictor,
                             weight_m_predictor, v_minus_act, num_bins, v_plus_predictor):
    # target_mask_predictor, label_map_predictor and spatial_weights_predictors are shape 1 x num_dist_bin 1 x 1

    x = np.arange(0, 10)
    fig, axis = plt.subplots(3, 1, figsize=(20, 20))
    target_predictor_initial = (init_gauss - init_gauss.min()).squeeze(0).squeeze(-1).cpu().numpy()
    target_predictor = target_map_predictor.weight.data.clone().detach().squeeze(0).squeeze(-1).cpu().numpy()
    axis[0].plot(x, target_predictor_initial, color='green')
    axis[0].plot(x, target_predictor, color='red')
    axis[0].set_title('target map predictor weights for y\', \ninitial weights in green min = {:0.2f}, max={:0.2f}'
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(target_predictor_initial), 2),
        np.round(np.amax(target_predictor_initial), 2),
        np.round(np.amin(target_predictor), 2),
        np.round(np.amax(target_predictor), 2)))

    weight_m_predictor_initial = init_weight_m_predictor
    weight_m_predictor = weight_m_predictor[0].weight.data.clone().detach()
    if v_minus_act == 'sigmoid':
        weight_m_predictor_initial = torch.sigmoid(weight_m_predictor_initial)
        weight_m_predictor = torch.sigmoid(weight_m_predictor)
    if v_minus_act == 'softplus':
        weight_m_predictor_initial = torch.nn.functional.softplus(weight_m_predictor_initial)
        weight_m_predictor = torch.nn.functional.softplus(weight_m_predictor)
    weight_m_predictor_initial = weight_m_predictor_initial.squeeze(0).squeeze(-1).cpu().numpy()
    weight_m_predictor = weight_m_predictor.squeeze(0).squeeze(-1).cpu().numpy()
    axis[1].plot(x, weight_m_predictor_initial, color='green')
    axis[1].plot(x, weight_m_predictor, color='red')
    axis[1].set_title('m predictor weights used for v_minus, \ninitial weights in green min = {:0.2f}, max={:0.2f} '
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(weight_m_predictor_initial), 2),
        np.round(np.amax(weight_m_predictor_initial), 2),
        np.round(np.amin(weight_m_predictor), 2),
        np.round(np.amax(weight_m_predictor), 2)))

    v_plus_predictor_initial = np.ones(num_bins)
    v_plus_predictor = v_plus_predictor.weight.data.clone().detach().squeeze(0).squeeze(
        -1).cpu().numpy()
    axis[2].plot(x, v_plus_predictor_initial, color='green')
    axis[2].plot(x, v_plus_predictor, color='red')
    axis[2].set_title('v_plus predictor weights, \ninitial weights in green min = {:0.2f}, max={:0.2f} '
                      '\nafter training in red min = {:0.2f}, max={:0.2f}'.format(
        np.round(np.amin(v_plus_predictor_initial), 2),
        np.round(np.amax(v_plus_predictor_initial), 2),
        np.round(np.amin(v_plus_predictor), 2),
        np.round(np.amax(v_plus_predictor), 2)))
    # hardcoded here
    fig.savefig('{}/{}.png'.format(save_dir,
                                   'local_weights_predictors'),
                bbox_inches='tight')
    plt.close(fig)
    print('plotted_weights')


def plot_correlation(target_image, source_image, flow_gt, correlation_volume, save_path, name, exchange_source_dimensions=False,
                     normalization='relu_l2norm'):
    '''

    :param target_image: 3xHxW
    :param source_image: 3xHxW
    :param correlation: (HxW)xH_c x W_c
    :return:
    '''
    # choose 10 random points
    _, H_ori, W_ori = flow_gt.shape
    _, H, W = correlation_volume.shape
    correlation_volume = correlation_volume.clone().detach()

    if exchange_source_dimensions:
        # the correlation is source dimension first !
        correlation_volume = correlation_volume.view(W, H, H, W).transpose(0, 1).contiguous().view(H * W, H, W)

    nbr_pts_per_row = 4
    X, Y = np.meshgrid(np.arange(4, W - 1, nbr_pts_per_row),
                       np.arange(4, H - 1, nbr_pts_per_row))
    X = X.flatten()
    Y = Y.flatten()

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)

    # resizing of source and target image to correlation size
    image_source = F.interpolate(source_image.unsqueeze(0).cpu() * std_values +
               mean_values, (H, W), mode='area', align_corners=False).squeeze().permute(1,2,0).numpy()
    image_target = F.interpolate(target_image.unsqueeze(0).cpu() * std_values +
               mean_values, (H, W), mode='area', align_corners=False).squeeze().permute(1,2,0).numpy()
    flow_gt_resized = F.interpolate(flow_gt.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0).cpu().numpy()
    flow_gt_resized[:,:,0] *= float(W)/float(W_ori)
    flow_gt_resized[:,:,1] *= float(H)/float(H_ori)

    fig, axis = plt.subplots(len(X), 3, figsize=(20, 20))
    for i in range(len(X)):
        pt = [X[i], Y[i]]
        # first coordinate is horizontal, second is vertical
        source_point = [int(round(pt[0] + flow_gt_resized[int(pt[1]), int(pt[0]), 0])),
                        int(round(pt[1] + flow_gt_resized[int(pt[1]), int(pt[0]), 1]))]
        correlation_at_point = correlation_volume.permute(1,2,0).view(H,W,H,W)[pt[1], pt[0]].cpu().numpy()
        max_pt_ = unravel_index(correlation_at_point.argmax(), correlation_at_point.shape)
        max_pt = [max_pt_[1], max_pt_[0]]

        min_pt_ = unravel_index(correlation_at_point.argmin(), correlation_at_point.shape)
        min_pt = [min_pt_[1], min_pt_[0]]

        axis[i][0].imshow(image_target, vmin=0, vmax=1.0)
        axis[i][0].scatter(pt[0],pt[1], s=4, color='red')
        axis[i][0].set_title('target_image')

        axis[i][1].imshow(image_source, vmin=0, vmax=1.0)
        if source_point[0] > W or source_point[0] < 0 or source_point[1] > H or source_point[1] < 0:
            axis[i][1].set_title('source_image, point is outside')
        else:
            axis[i][1].scatter(source_point[0],source_point[1], s=4, color='red')
            axis[i][1].set_title('source_image')

        min_value = np.amin(correlation_at_point)
        max_value = np.amax(correlation_at_point)
        if normalization == 'softmax' or normalization == 'relu_l2norm':
            axis_min_value = 0.0
            axis_max_value = 1.0
        else:
            axis_min_value = -1.0
            axis_max_value = 1.0
        im1 = axis[i][2].imshow(correlation_at_point, vmin=axis_min_value, vmax=axis_max_value)
        axis[i][2].scatter(max_pt[0],max_pt[1], s=4, color='red')
        axis[i][2].scatter(min_pt[0],min_pt[1], s=4, color='green')
        axis[i][2].set_title('Global correlation, max is {} (red), min is {} (green)'.format(
            max_value, min_value ))
        fig.colorbar(im1, ax=axis[i][2])
    fig.savefig('{}/{}.png'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)
