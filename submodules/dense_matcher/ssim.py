#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import torch
import torch.nn.functional as F
import numpy as np
from math import exp, sqrt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, std=1.5):
    _1D_window = gaussian(window_size, std).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(pred, gt, window, channel):
    ntotpx, nviews, nc, h, w = pred.shape
    flat_pred = pred.view(-1, nc, h, w)
    mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc)
    mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2).unsqueeze(1)
    mu1_mu2 = mu1 * mu2.unsqueeze(1)

    sigma1_sq = F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, 3) - mu2_sq
    sigma12 = F.conv2d((pred * gt.unsqueeze(1)).view(-1, nc, h, w), window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    similarity = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    values = 1 - similarity
    return torch.sum(values, dim=2) / 3
    # return similarity

class SSIM(torch.nn.Module):
    def __init__(self, h_patch_size):
        super(SSIM, self).__init__()
        self.window_size = 2 * h_patch_size + 1
        self.channel = 3
        self.register_buffer("window", create_window(self.window_size, self.channel))

    def forward(self, img_pred, img_gt):
        nviews, ntotpx, npatch, channels = img_pred.shape

        patch_size = int(sqrt(npatch))
        patch_img_pred = img_pred.reshape(nviews, ntotpx, patch_size, patch_size, channels).permute(1, 0, 4, 2, 3).contiguous()
        patch_img_gt = img_gt.reshape(ntotpx, patch_size, patch_size, channels).permute(0, 3, 1, 2)

        return _ssim(patch_img_pred, patch_img_gt, self.window, self.channel)



def _ssim_v2(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = 0, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = 0, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map

class SSIM_v2(torch.nn.Module):
    def __init__(self, h_patch_size=5):
        super(SSIM_v2, self).__init__()
        self.window_size = 2 * h_patch_size + 1
        self.channel = 3
        self.register_buffer("window", create_window(self.window_size, self.channel))

    def forward(self, img_pred, img_gt):
        n_view, N, n_p, npatch, channels = img_pred.shape
        patch_size = int(sqrt(npatch))
        patch_img_pred = img_pred.reshape(n_view, N, n_p, patch_size, patch_size, channels).permute(0, 1, 2, 5, 3, 4)
        patch_img_gt = img_gt.reshape(1, N, -1, patch_size, patch_size, channels).permute(0, 1, 2, 5, 3, 4)
        patch_img_gt = patch_img_gt.expand(n_view, N, n_p, channels, patch_size, patch_size)

        patch_img_pred = patch_img_pred.reshape(n_view*N*n_p, channels, patch_size, patch_size)
        patch_img_gt = patch_img_gt.reshape(n_view*N*n_p, channels, patch_size, patch_size)

        ssim = _ssim_v2(patch_img_pred, patch_img_gt, self.window, self.window_size, channels)

        ssim = ssim.reshape(n_view, N, n_p, channels).mean(-1)
        return ssim
