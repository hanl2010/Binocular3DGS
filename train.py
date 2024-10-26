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

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, SmoothLoss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import inverse_warp_images
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    # Create tensors for binocular consistency
    image_height = viewpoint_stack[0].image_height
    image_width = viewpoint_stack[0].image_width
    row_indices = torch.arange(0, image_height).view(-1, 1).repeat(1, image_width).cuda()
    column_indices = torch.arange(0, image_width).repeat(image_height, 1).cuda()
    mask = torch.ones((1, image_height, image_width), dtype=torch.float32).cuda()

    smooth_loss = SmoothLoss()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # random_idx = randint(0, len(viewpoint_stack)-1)
        # viewpoint_cam = viewpoint_stack[random_idx]
        viewpoint_cam = random.choice(viewpoint_stack)

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["rendered_depth"]
        alpha = render_pkg["rendered_alpha"]

        gt_image = viewpoint_cam.original_image.cuda()

        ######## from DNGaussian ###########
        bg_mask = None
        if args.dataset_name == 'DTU':
            if 'scan110' not in args.source_path:
                bg_mask = (gt_image.max(0, keepdim=True).values < 30 / 255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15 / 255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            bg_mask = bg_mask.float()
        #####################################

        disparity_loss=0.0
        if args.binocular_consistency and iteration > args.shift_cam_start:
            trans_dist = torch.rand(1) * 0.4
            trans_dist = (trans_dist * random.choice([-1.0, 1.0])).item()
            shifted_cam = scene.getShiftedCamera(viewpoint_cam, trans_dist)
            render_pkg = render(shifted_cam, gaussians, pipe, bg)
            shifted_image = render_pkg["render"]
            focal_x, focal_y = viewpoint_cam.get_focal()
            disparity = focal_x * (-trans_dist) / (depth + 1e-5)
            warped_image = inverse_warp_images(shifted_image.unsqueeze(0), disparity.unsqueeze(0), row_indices, column_indices)
            shift_mask = inverse_warp_images(mask.unsqueeze(0), disparity.unsqueeze(0), row_indices, column_indices)

            disparity_loss = (l1_loss(warped_image, gt_image.unsqueeze(0), mask=shift_mask) +
                              0.05 * smooth_loss.forward(disparity=disparity*shift_mask, image=gt_image.unsqueeze(0)))


        alpha_loss = 0.0
        if viewpoint_cam.gt_alpha_mask is not None:
            alpha_loss = torch.mean(torch.abs(alpha) * (1-viewpoint_cam.gt_alpha_mask))
        elif bg_mask is not None:
            alpha_loss = torch.mean(torch.abs(alpha) * bg_mask)

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        total_loss = loss + disparity_loss + alpha_loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.6f}",
                                          "disp_loss": f"{disparity_loss:.6f}",
                                          "points": f"{len(gaussians.get_xyz)}",
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (pipe, background))
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if args.opacity_decay and iteration > opt.densify_from_iter:
                opt.densify_until_iter = opt.iterations
                gaussians.opacity_decay(factor=args.opacity_decay_factor)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if args.opacity_decay:
                        size_threshold = None
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    pass
                    # if args.opacity_decay:
                    #     pass
                    # else:
                    #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--opacity_decay", action="store_true", default=True)
    parser.add_argument("--opacity_decay_factor", type=float, default=0.995)
    parser.add_argument("--binocular_consistency", action="store_true", default=True)
    parser.add_argument("--shift_cam_start", type=int, default=20000)
    parser.add_argument("--dataset_name", type=str, default="LLFF")
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete." + args.model_path)
