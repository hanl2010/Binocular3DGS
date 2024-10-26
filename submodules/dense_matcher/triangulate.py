import cv2
import torch
import os
import numpy as np
import imageio
from torchvision import transforms
from tqdm import tqdm
import trimesh
import torch.nn.functional as F

from colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from model_selection import model_type, pre_trained_model_types, select_model
from utils import DotDict, matches_from_flow, getView2World, point_world2depth, depth2point_world, get_projected_patch_color, map_points_to_image
from ssim import SSIM_v2



from argparse import ArgumentParser
parser = ArgumentParser(description="Triangulate script parameters")
parser.add_argument("--network_type", type=str, default="PDCNet_plus")
parser.add_argument("--pre_trained_model", type=str, default="megadepth")
parser.add_argument("--multi_stage_type", type=str, choices=['d', 'h', 'ms'], default="h")
parser.add_argument("--confidence_map_R", type=float, default=1.0)
parser.add_argument("--ransac_thresh", type=float, default=1.0)
parser.add_argument("--mask_type", type=str, default="proba_interval_1_above_10")
parser.add_argument("--homography_visibility_mask", action="store_true", default=True)
parser.add_argument("--scaling_factors", type=float, nargs="+", default=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2])
parser.add_argument("--compute_cyclic_consistency_error", action="store_true", default=True)

parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--n_views", type=int, default=3)
parser.add_argument("--resolution", type=int, default=4)
parser.add_argument("--dtu_sparse_indices", type=int, nargs="+", default=[25, 22, 28, 40, 44, 48, 0, 8, 13])
parser.add_argument("--output_path", type=str, default="keypoints_to_3d")
parser.add_argument("--dataset_name", type=str, default="LLFF")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

if args.network_type not in model_type:
    raise ValueError('The model that you chose is not valid: {}'.format(args.network_type))
if args.pre_trained_model not in pre_trained_model_types:
    raise ValueError('The pre-trained model type that you chose is not valid: {}'.format(args.pre_trained_model))
choices_for_multi_stage_types = ['d', 'h', 'ms']
if args.multi_stage_type not in choices_for_multi_stage_types:
    raise ValueError('The inference mode that you chose is not valid: {}'.format(args.multi_stage_type))

global_optim_iter = 3
local_optim_iter = 7
path_to_pre_trained_models = 'pre_trained_models/'
matcher, estimate_uncertainty = select_model(args.network_type, args.pre_trained_model, args, global_optim_iter, local_optim_iter,
                                             path_to_pre_trained_models=path_to_pre_trained_models)

scene_name = os.path.basename(args.data_path)
image_fold = "images"
colmap_camera_bin = os.path.join(args.data_path, "sparse/0/cameras.bin")
colmap_images_bin = os.path.join(args.data_path, "sparse/0/images.bin")

# read cameras
cam_intrinsics = read_intrinsics_binary(colmap_camera_bin)
cam_extrinsics = read_extrinsics_binary(colmap_images_bin)
intrinsics_all = []
extrinsics_all = []
images_list = []
for key in sorted(cam_extrinsics.keys()):
    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    height = intr.height
    width = intr.width
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)
    c2w = getView2World(R, T)
    extrinsics_all.append(c2w)
    images_list.append(os.path.join(args.data_path, image_fold, extr.name))

    if intr.model == "SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
        center_x = width / 2
        center_y = height / 2
    elif intr.model == "PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        center_x = intr.params[2]
        center_y = intr.params[3]
    else:
        raise NotImplementedError(intr.model)

    intrinsic = np.zeros((3,3))
    intrinsic[0, 0] = focal_length_x / args.resolution
    intrinsic[1, 1] = focal_length_y / args.resolution
    intrinsic[0, 2] = center_x / args.resolution
    intrinsic[1, 2] = center_y / args.resolution
    intrinsic[2, 2] = 1.0
    intrinsics_all.append(intrinsic)

extrinsics_all = np.stack(extrinsics_all)
intrinsics_all = np.stack(intrinsics_all)
extrinsics_all = torch.tensor(extrinsics_all).float().cuda()
intrinsics_all = torch.tensor(intrinsics_all).float().cuda()


n_images = len(images_list)
if args.dataset_name == "LLFF":
    llffhold=8
    train_idx = [idx for idx in range(n_images) if idx % llffhold != 0]
    idx_sub = [round(i) for i in np.linspace(0, len(train_idx) - 1, args.n_views)]
elif args.dataset_name == "DTU":
    idx_sub = args.dtu_sparse_indices[:args.n_views]
else:
    raise NotImplementedError(args.dataset_name)
ref_indices = idx_sub
srcs_indices = {}
for idx in ref_indices:
    indices = ref_indices.copy()
    indices.remove(idx)
    srcs_indices[idx] = indices

print("reading images ...")
images = np.stack([imageio.v3.imread(image_name)[..., :3] for image_name in tqdm(images_list)])
images = torch.tensor(images).float().cuda()
if args.resolution > 1:
    height = height // args.resolution
    width = width // args.resolution
    images = transforms.Resize(size=(height, width), antialias=True)(images.permute(0, 3, 1, 2))
    images = images.permute(0, 2, 3, 1)

# focal = torch.tensor([intrinsic[0, 0], intrinsic[1, 1]], device="cuda")
# center = torch.tensor([intrinsic[0, 2], intrinsic[1, 2]], device="cuda")
image_wh = torch.tensor([width - 1, height - 1], device="cuda")
image_w = width
image_h = height

print("start key points prediction")
points_3D = []
colors_all = []
for ref_index in ref_indices:
    depth_list = []
    coord_list = []
    for src_index in srcs_indices[ref_index]:
        ref_image = images[ref_index]
        src_image = images[src_index]

        with torch.inference_mode():
            pred = matcher.get_matches_and_confidence(ref_image.permute(2,0,1).unsqueeze(0),
                                                      src_image.permute(2,0,1).unsqueeze(0))
        mkpts0 = pred["kp_source"]
        mkpts1 = pred["kp_target"]
        confidence = pred["confidence_value"]
        # mask = (confidence > 0.2)
        # mkpts0 = mkpts0[mask]
        # mkpts1 = mkpts1[mask]

        ref_image_name = os.path.basename(images_list[ref_index])
        ref_image_name = ref_image_name.split(".")[0]
        src_image_name = os.path.basename(images_list[src_index])
        src_image_name = src_image_name.split(".")[0]
        print(f"key points from {ref_image_name} and {src_image_name} : {len(mkpts0)}")
        if len(mkpts0) == 0:
            continue

        # Use triangulation to get 3D points
        ref_c2w = extrinsics_all[ref_index]
        src_c2w = extrinsics_all[src_index]
        intrinsic = torch.cat([intrinsics_all[0], torch.zeros((3, 1), device="cuda")], dim=1)
        ref_p = torch.matmul(intrinsic, torch.inverse(ref_c2w))
        src_p = torch.matmul(intrinsic, torch.inverse(src_c2w))
        points = cv2.triangulatePoints(ref_p.cpu().numpy(), src_p.cpu().numpy(), mkpts0.T, mkpts1.T)
        points = (points / points[3])[:3].T

        # project 3D points to image
        points = torch.tensor(points, device="cuda")
        ref_w2c = torch.inverse(ref_c2w)
        src_w2c = torch.inverse(src_c2w)
        ref_uv, ref_depth = point_world2depth(points=points.reshape(-1, 3),
                                      intrinsic_matrix=intrinsic[:3, :3],
                                      w2c=ref_w2c)
        src_uv, src_depth = point_world2depth(points=points.reshape(-1, 3),
                                      intrinsic_matrix=intrinsic[:3, :3],
                                      w2c=src_w2c)

        ### filter ####
        mkpts0_tensor = torch.tensor(mkpts0).cuda()
        mkpts1_tensor = torch.tensor(mkpts1).cuda()
        ref_norm = torch.norm(ref_uv - mkpts0_tensor, dim=-1)
        src_norm = torch.norm(src_uv - mkpts1_tensor, dim=-1)

        mask = (ref_norm < 2.0) & (src_norm < 2.0)
        points = points[mask]
        ref_uv = ref_uv[mask]
        src_uv = src_uv[mask]
        ref_depth = ref_depth[mask]
        src_depth = src_depth[mask]
        mkpts0_tensor = mkpts0_tensor[mask]
        mkpts1_tensor = mkpts1_tensor[mask]


        uv_mask = (ref_uv[:, 0] >= 0) & (ref_uv[:, 0] <= image_w-1) & (ref_uv[:, 1] >= 0) & (ref_uv[:, 1]<=image_h-1) & \
                  (src_uv[:, 0] >= 0) & (src_uv[:, 0] <= image_w-1) & (src_uv[:, 1] >= 0) & (src_uv[:, 1]<=image_h-1)
        points = points[uv_mask]
        ref_uv = ref_uv[uv_mask]
        src_uv = src_uv[uv_mask]
        ref_depth = ref_depth[uv_mask]
        src_depth = src_depth[uv_mask]
        mkpts0_tensor = mkpts0_tensor[uv_mask]
        mkpts1_tensor = mkpts1_tensor[uv_mask]


        points_3D.append(points.cpu().numpy())

        ref_uv_normal = (ref_uv / image_wh) * 2 - 1.0
        colors = F.grid_sample(ref_image.permute(2, 0, 1).unsqueeze(0), grid=ref_uv_normal.reshape(1, -1, 1, 2),
                               align_corners=False)
        colors = colors[0, :, :, 0].permute(1, 0)
        colors = colors.cpu().numpy()
        colors_all.append(colors.astype(np.uint8))

    if args.dataset_name == "DTU":
        depth_max = 10.0
        image = images[ref_index]
        intrinsic = intrinsics_all[ref_index]
        extrinsic = extrinsics_all[ref_index]
        depth = torch.ones_like(image, device="cuda")[..., 0] * depth_max

        depth_points = depth2point_world(depth, intrinsic, torch.inverse(extrinsic))
        depth_points = depth_points.cpu().numpy()
        # depth_colors = image.reshape(-1, 3).numpy().astype(np.uint8)
        depth_colors = (torch.ones_like(image, device="cpu").reshape(-1, 3).numpy() * 255.0).astype(np.uint8)

        bg_mask = (image.max(dim=-1, keepdim=True).values >= 254).reshape(-1).cpu().numpy()
        depth_points = depth_points[bg_mask]
        depth_colors = depth_colors[bg_mask]

        points_3D.append(depth_points)
        colors_all.append(depth_colors)


points_3D = np.concatenate(points_3D, axis=0)
colors_all = np.concatenate(colors_all, axis=0)

mesh = trimesh.Trimesh(vertices=points_3D, vertex_colors=colors_all)
mesh.export(os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d.ply"))

if args.dataset_name == "LLFF":
    iterations = 1000
    ssim_threshold = 0.95
    h_patch_size = 5
    save_iteration = 1000
    alpha = 10.0  # 随机采样的范围系数
    ssim_func = SSIM_v2(h_patch_size=h_patch_size).cuda()
    intrinsic = intrinsics_all[0]
    focal = torch.tensor([intrinsic[0, 0], intrinsic[1, 1]], device="cuda")
    center = torch.tensor([intrinsic[0, 2], intrinsic[1, 2]], device="cuda")
    image_wh = torch.tensor([width - 1, height - 1], device="cuda")
    # lcf = LocalOutlierFactor(n_neighbors=50)

    print("starting sample points ...")
    points_all = torch.tensor(points_3D).float().cuda()
    colors_all = torch.tensor(colors_all).float().cuda()
    for iteration in tqdm(range(iterations)):
        iteration = iteration + 1
        ref_idx = ref_indices[torch.randperm(len(ref_indices))[0]]
        src_len = len(srcs_indices[ref_idx])
        src_idx = srcs_indices[ref_idx][torch.randperm(src_len)[0]]

        ref_image = images[ref_idx] / 255.0
        src_image = images[src_idx] / 255.0

        ref_c2w = extrinsics_all[ref_idx]
        src_c2w = extrinsics_all[src_idx]

        ref_mask = torch.ones(ref_image.shape[:2], device="cuda")
        src_mask = torch.ones(src_image.shape[:2], device="cuda")

        ## random sample in pcd
        sample_points_num = 100
        sample_points = points_all[torch.randperm(len(points_3D))[:sample_points_num]]

        ## Generate random sampling points
        sample_num = 200
        rand_p = torch.randn(size=(sample_points_num, sample_num, 3), device="cuda")
        rand_p = sample_points[:, None, :] + rand_p * alpha

        # pcd_all = np.concatenate([positions, rand_p.reshape(-1, 3)])
        # if iteration % save_iteration == 0:
        #     points = points_all.cpu().numpy()
        #     colors = colors_all.cpu().numpy().astype(np.uint8)
        #     mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors)
        #     mesh.export(os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d_sample_{iteration}.ply"))
        #     mesh.export(os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d.ply"))
        #     print("export:", os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d_sample_{iteration}.ply"))

        ## the random sampled points are projected to ref_image and src_image, and get patches
        rand_sample_points = rand_p.reshape(-1, 1, 3)
        patch_sample = get_projected_patch_color(sample_points=rand_sample_points,
                                                 ref_image=ref_image,
                                                 src_images=src_image.unsqueeze(0),
                                                 ref_c2w=ref_c2w,
                                                 src_c2ws=src_c2w.unsqueeze(0),
                                                 intrinsic=intrinsic,
                                                 h_patch_size=h_patch_size,
                                                 )
        ref_patch = patch_sample["ref_patch"]
        src_patch = patch_sample["src_patch"]
        patch_mask = patch_sample["patch_mask"]
        n_view, N, n_p = src_patch.shape[:3]

        ssim = ssim_func.forward(src_patch, ref_patch.unsqueeze(2))
        ssim = ssim * patch_mask
        ssim = torch.mean(ssim, dim=0).squeeze()

        selected = (ssim >= ssim_threshold)
        new_points = rand_sample_points.squeeze()[selected]

        if len(new_points) > 0:
            # new_points = new_points.cpu().numpy()
            old_points_num = len(points_all)
            new_points_num = len(new_points)
            # points_tmp = np.concatenate([points_all, new_points], axis=0)
            points_tmp = torch.cat([points_all, new_points], dim=0)

            ref_uv = map_points_to_image(points=points_tmp.reshape(-1, 1, 3),
                                         src_poses_inv=torch.inverse(ref_c2w).unsqueeze(0),
                                         focal=focal,
                                         center=center)
            ref_uv = ref_uv.squeeze()
            ref_uv_round = torch.round(ref_uv)

            # Filter out the points that are outside the ref_mask
            ref_uv_new = ref_uv[-new_points_num:]
            ref_uv_new_normal = (ref_uv_new / image_wh) * 2 - 1.0
            points_mask = F.grid_sample(ref_mask[None, None, ...], grid=ref_uv_new_normal.reshape(1, -1, 1, 2),
                                        align_corners=False)
            points_mask = points_mask.squeeze().bool()
            if points_mask.sum() == 0:
                continue

            unique_uv, reverse_index, count = torch.unique(ref_uv_round, return_inverse=True, return_counts=True,
                                                           dim=0)
            count_all = count[reverse_index]
            count_new = count_all[-new_points_num:]
            ref_index_selected = points_mask & (count_new <= 2)

            new_colors = F.grid_sample(ref_image.permute(2, 0, 1).unsqueeze(0),
                                       grid=ref_uv_new_normal.reshape(1, -1, 1, 2),
                                       align_corners=False)
            new_colors = new_colors.squeeze(0).squeeze(-1).permute(1, 0)

            src_uv = map_points_to_image(points=points_tmp.reshape(-1, 1, 3),
                                         src_poses_inv=torch.inverse(src_c2w).unsqueeze(0),
                                         focal=focal,
                                         center=center)
            src_uv = src_uv.squeeze()
            src_uv_round = torch.round(src_uv)
            # Filter out the points that are outside the src_mask
            src_uv_new = src_uv[-new_points_num:]
            src_uv_new_normal = (src_uv_new / image_wh) * 2 - 1.0
            points_mask = F.grid_sample(src_mask[None, None, ...], grid=src_uv_new_normal.reshape(1, -1, 1, 2),
                                        align_corners=False)
            points_mask = points_mask.squeeze().bool()
            if points_mask.sum() == 0:
                continue

            unique_uv, reverse_index, count = torch.unique(src_uv_round, return_inverse=True, return_counts=True,
                                                           dim=0)
            count_all = count[reverse_index]
            count_new = count_all[-new_points_num:]
            src_index_selected = points_mask & (count_new <= 2)

            index_selected = ref_index_selected & src_index_selected

            new_points_selected = new_points[index_selected]
            new_colors_selected = (new_colors[index_selected] * 255.0)
            points_all = torch.cat([points_all, new_points_selected])
            colors_all = torch.cat([colors_all, new_colors_selected])
            positions = points_all

    points = points_all.cpu().numpy()
    colors = colors_all.cpu().numpy().astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors)
    # mesh.export(os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d_sample_{iteration}.ply"))
    mesh.export(os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d.ply"))
    print("export:", os.path.join(args.output_path, f"{scene_name}_keypoints_to_3d.ply"))