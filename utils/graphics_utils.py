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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

## from https://github.com/mrharicot/monodepth
def inverse_warp_images(image, disparity, row_indices, column_indices):
    r_ind = row_indices
    c_ind = column_indices
    # For bilinear interp. between two sets of pixel offsets (since disparities are floats that fall inbetween pixels)
    x0 = torch.floor(disparity).type(torch.LongTensor).cuda()
    x1 = x0 + 1

    # Empty list to store warped batch images
    warped_ims = []

    # How to do this without loops???
    for b in range(image.size(0)):  # Loop over images in batch

        # Empty list to store warped channels for current image
        warped_ims_ch = []

        for ch in range(image.size(1)):  # Loop over RGB channels

            # Column indicies for left side of bilinear interpolation
            c_ind_d_0 = c_ind + x0[b, 0]

            # Mask of invalid indices
            c_ind_d_0_invalid = (c_ind_d_0 < 0) | (c_ind_d_0 >= image.size(3))

            # Make indices within bounds
            c_ind_d_0[c_ind_d_0 >= image.size(3)] = image.size(3) - 1
            c_ind_d_0[c_ind_d_0 < 0] = 0

            # Same as above but for right side of interpolation
            c_ind_d_1 = c_ind + x1[b, 0]
            c_ind_d_1_invalid = (c_ind_d_1 < 0) | (c_ind_d_1 >= image.size(3))
            c_ind_d_1[c_ind_d_1 >= image.size(3)] = image.size(3) - 1
            c_ind_d_1[c_ind_d_1 < 0] = 0

            # Inverse warp
            warped_ims_ch.append(((x1[b, 0] - disparity[b, 0]) * image[b, ch, r_ind, c_ind_d_0] + (
                        disparity[b, 0] - x0[b, 0]) * image[b, ch, r_ind, c_ind_d_1]).unsqueeze(0).unsqueeze(0))

            # Set invalid areas to 0
            warped_ims_ch[-1][0, 0, c_ind_d_0_invalid] = 0.0
            warped_ims_ch[-1][0, 0, c_ind_d_1_invalid] = 0.0

        # List to tensor
        warped_ims.append(torch.cat(warped_ims_ch, 1))

    return torch.cat(warped_ims, 0)