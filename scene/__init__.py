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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos
from scene.cameras import Camera
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, args.dataset_name, suffix=args.suffix)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views, args.dataset_name)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getShiftedCamera(self, camera, trans_dist=0.1):
        intrinsic, extrinsic = camera.get_camera_matrix()
        point = torch.tensor([trans_dist, 0.0, 0.0, 1.0], device="cuda")
        point_world = torch.inverse(extrinsic) @ point
        point_world = point_world[:3]
        camera_center_trans = (point_world - camera.camera_center).cpu().numpy()
        camera = Camera(
            colmap_id=camera.colmap_id,
            R=camera.R,
            T=camera.T,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            image=torch.ones_like(camera.original_image),
            gt_alpha_mask=None,
            image_name=None,
            uid=camera.uid,
            trans=camera_center_trans,
            data_device=camera.data_device
        )
        return camera

    def getInterpolatedCamera(self, camera, camera_src):
        intrinsics, extrinsic = camera.get_calib_matrix_nerf()
        intrinsics, extrinsic_src = camera_src.get_calib_matrix_nerf()
        c2w = torch.inverse(extrinsic)
        c2w_src = torch.inverse(extrinsic_src)
        w = torch.rand(1).squeeze()
        c2w_unseen = w * c2w + (1-w)*c2w_src
        w2c_unseen = torch.inverse(c2w_unseen)
        R = w2c_unseen[:3, :3].transpose(0, 1)
        T = w2c_unseen[:3, 3]
        camera = Camera(
            colmap_id=camera.colmap_id,
            R = R.cpu().numpy(),
            T = T.cpu().numpy(),
            FoVx = camera.FoVx,
            FoVy = camera.FoVy,
            image = torch.ones_like(camera.original_image),
            gt_alpha_mask=None,
            image_name=None,
            uid=None,
            data_device=camera.data_device
        )
        return camera


class RenderScene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, spiral=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        if 'scan' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
        else:
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Render Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras,
                                                                                 resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            pass

    def getRenderCameras(self, scale=1.0):
        return self.test_cameras[scale]