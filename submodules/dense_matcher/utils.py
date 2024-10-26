import torch
import numpy as np
import torch.nn.functional as F

class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


# --- MATCHES FROM FLOW UTILS ---
def matches_from_flow(flow, binary_mask, scaling=1.0):
    """
    Retrieves the pixel coordinates of 'good' matches in source and target images, based on provided flow field
    (relating the target to the source image) and a binary mask indicating where the flow is 'good'.
    Args:
        flow: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating the target
              to the source image, defined in the target image coordinate system.
        binary_mask: bool mask corresponding to valid flow vectors, shape B, H, W
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.

    Returns:
        pixel coordinates of 'good' matches in the source image, Nx2 (numpy array)
        pixel coordinates of 'good' matches in the target image, Nx2 (numpy array)
    """

    B, _, hB, wB = flow.shape
    xx = torch.arange(0, wB).view(1, -1).repeat(hB, 1)
    yy = torch.arange(0, hB).view(-1, 1).repeat(1, wB)
    xx = xx.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flow.is_cuda:
        grid = grid.cuda()
        binary_mask = binary_mask.cuda()

    mapping = flow + grid
    mapping_x = mapping.permute(0, 2, 3, 1)[:, :, :, 0]
    mapping_y = mapping.permute(0, 2, 3, 1)[:, :, :, 1]
    grid_x = grid.permute(0, 2, 3, 1)[:, :, :, 0]
    grid_y = grid.permute(0, 2, 3, 1)[:, :, :, 1]

    pts2 = torch.cat((grid_x[binary_mask].unsqueeze(1),
                      grid_y[binary_mask].unsqueeze(1)), dim=1)
    pts1 = torch.cat((mapping_x[binary_mask].unsqueeze(1),
                      mapping_y[binary_mask].unsqueeze(1)),
                     dim=1)  # convert to mapping and then take the correspondences

    return pts1.cpu().numpy()*scaling, pts2.cpu().numpy()*scaling

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getView2World(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(np.linalg.inv(Rt))

def point_world2depth(points, intrinsic_matrix, w2c):
    points_rot = torch.matmul(w2c[None, :3, :3], points[:, :, None])[..., 0]
    points_trans = points_rot + w2c[None, :3, 3]  # (n_p, 3)
    points_image = torch.matmul(intrinsic_matrix, points_trans[..., None])[..., 0]
    uv = points_image[:, :2] / points_image[:, 2:]
    depth = points_image[:, 2:]
    # focal = torch.tensor([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]] ,device="cuda")
    # center = torch.tensor([intrinsic_matrix[0,2], intrinsic_matrix[1,2]] ,device="cuda")
    return uv, depth

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=intrinsic.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz
def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], dim=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]
    return xyz_world

def get_projected_patch_color(sample_points, ref_image, src_images, ref_c2w, src_c2ws,
                              intrinsic,
                              h_patch_size):
    # ref_image = ref_view_cam.original_image
    # intrinsic, ref_w2c = ref_view_cam.get_calib_matrix_nerf()

    # src_images = []
    # src_w2cs = []
    # for cam in src_view_cams:
    #     src_images.append(cam.original_image)
    #     intrinsic, src_w2c = cam.get_calib_matrix_nerf()
    #     src_w2cs.append(src_w2c)
    # src_images = torch.stack(src_images, dim=0)
    # src_w2cs = torch.stack(src_w2cs, dim=0)
    ref_w2c = torch.inverse(ref_c2w)
    src_w2cs = torch.inverse(src_c2ws)

    image_h = ref_image.shape[0]
    image_w = ref_image.shape[1]
    image_wh = torch.tensor([image_w, image_h], device="cuda")

    ref_image = ref_image.permute(2, 0, 1)
    src_images = src_images.permute(0, 3, 1, 2)
    sample = {}

    focal = torch.tensor([intrinsic[0, 0], intrinsic[1, 1]], device="cuda")
    center = torch.tensor([intrinsic[0, 2], intrinsic[1, 2]], device="cuda")

    src_uv = map_points_to_image(sample_points, src_w2cs, focal, center)
    n_view, N, n_p, _ = src_uv.shape
    src_mask = (src_uv[..., 0] >= 0) & (src_uv[..., 0] < image_w) & (src_uv[..., 1] >= 0) & (src_uv[..., 1] < image_h)

    offset = build_patch_offset(half_patch_size=h_patch_size)
    grid = src_uv.reshape(n_view, N * n_p, 1, 2) + offset[None, None, :, :]
    grid_normal = grid * 2 / image_wh - 1.0  # (-1, 1)
    src_patch = F.grid_sample(src_images, grid_normal, align_corners=False)
    src_patch = src_patch.permute(0, 2, 3, 1).reshape(n_view, N, n_p, -1, 3)

    ref_uv = map_points_to_image(sample_points, ref_w2c.unsqueeze(0), focal, center)
    ref_mask = (ref_uv[..., 0] >= 0) & (ref_uv[..., 0] < image_w) & (ref_uv[..., 1] >= 0) & (ref_uv[..., 1] < image_h)
    grid = ref_uv.reshape(1, N, 1, 2) + offset[None, None, :, :]
    grid_normal = grid * 2 / image_wh - 1.0
    ref_patch = F.grid_sample(ref_image.unsqueeze(0), grid_normal, align_corners=False)
    ref_patch = ref_patch.permute(0, 2, 3, 1).reshape(1, N, -1, 3)

    mask = ref_mask & src_mask
    sample["src_patch"] = src_patch
    sample["ref_patch"] = ref_patch
    sample["patch_mask"] = mask

    return sample

def map_points_to_image(points, src_poses_inv, focal, center):
    """
            把光线上的点映射到图像上,得到图像上的坐标
            points: 光线上的点 （N, n_p, 3）
            src_poses_inv: 源图像的相机姿态的逆 (n_views, 4, 4)
            focal: 相机焦距 (2, ) (fx, fy)
            center: 图像中心 (2, ) (cx, cy)
            return: uv (n_view, N, n_p, 2)
            """
    # src_poses_inv = torch.inverse(src_poses)  # 得到姿态矩阵的逆
    points_rot = torch.matmul(src_poses_inv[:, None, None, :3, :3], points[None, :, :, :, None])[..., 0]
    points_trans = points_rot + src_poses_inv[:, None, None, :3, 3]  # (n_view, N, n_p, 3)
    uv = points_trans[..., :2] / points_trans[..., 2:]  # 转为Z轴坐标归一化的点， 与get_rays的过程相反
    uv *= focal[None, None, None, :]  # (n_view, N, n_p, 2)
    uv += center[None, None, None, :]  # (n_view, N, n_p, 2) # 得到图像中的像素坐标
    return uv

def build_patch_offset(half_patch_size=5):
    assert half_patch_size > 0, "half_patch_size error: {}".format(half_patch_size)
    offset_range = torch.arange(-half_patch_size, half_patch_size+1, device="cuda")
    offset_y, offset_x = torch.meshgrid(offset_range, offset_range)
    offset = torch.stack([offset_x, offset_y], dim=-1).reshape(-1, 2)
    return offset.float()