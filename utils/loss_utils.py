#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn

def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return torch.abs((network_output*mask - gt*mask)).mean()
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.edge_conv_x_3 = torch.nn.Conv2d(3, 1, 3, bias=False).cuda()
        self.edge_conv_y_3 = torch.nn.Conv2d(3, 1, 3, bias=False).cuda()
        self.edge_conv_x_1 = torch.nn.Conv2d(1, 1, 3, bias=False).cuda()
        self.edge_conv_y_1 = torch.nn.Conv2d(1, 1, 3, bias=False).cuda()

        # Set layer weights to be edge filters
        with torch.no_grad():
            for layer in [self.edge_conv_x_3, self.edge_conv_x_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0, ch] = torch.Tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).cuda()

            for layer in [self.edge_conv_y_3, self.edge_conv_y_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0, ch] = torch.Tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]).cuda()

    def forward(self, disparity, image):
        edge_x_im = torch.exp((self.edge_conv_x_3(image).abs() * -0.33))
        edge_y_im = torch.exp((self.edge_conv_y_3(image).abs() * -0.33))
        edge_x_d = self.edge_conv_x_1(disparity)
        edge_y_d = self.edge_conv_y_1(disparity)
        return ((edge_x_im * edge_x_d)).abs().mean() + ((edge_y_im * edge_y_d)).abs().mean()