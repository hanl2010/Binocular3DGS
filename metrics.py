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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import torchvision

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, dataset_name, idrmasks_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # read mask
                if dataset_name == "DTU":
                    print("eval DTU with mask ...")
                    masks = []
                    dtu_test_indices = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 43, 45,
                                46, 47]
                    scene = os.path.basename(scene_dir).split("_")[0]
                    image_h, image_w = renders[0].shape[2:]
                    for image_name in image_names:
                        idx = int(image_name.split(".")[0])
                        mask_name = f"{dtu_test_indices[idx]:03d}.png"
                        if scene in ["scan110", "scan114", "scan40", "scan55", "scan63"]:
                            mask_image_path = os.path.join(idrmasks_path, scene, "mask", mask_name)
                        else:
                            mask_image_path = os.path.join(idrmasks_path, scene, mask_name)
                        mask_image = Image.open(mask_image_path)
                        mask_image = mask_image.resize((image_w, image_h))
                        mask = tf.to_tensor(mask_image)[:3].cuda()
                        masks.append(mask)
                else:
                    masks = torch.ones(size=(len(renders), *renders[0].shape[1:])).float().cuda()

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render = renders[idx] * masks[idx] + (1 - masks[idx])
                    gt = gts[idx] * masks[idx] + (1 - masks[idx])

                    if method == "ours_30000":
                        os.makedirs(method_dir / "masked", exist_ok=True)
                        torchvision.utils.save_image(render, os.path.join(method_dir / "masked",
                                                                          '{0:05d}'.format(idx) + ".png"))

                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(render, gt, mask=masks[idx].unsqueeze(0)))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--idrmasks_path", type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, args.dataset_name, args.idrmasks_path)
