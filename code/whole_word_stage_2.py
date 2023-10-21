from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import (
    SDSLoss, 
    ToneLoss, 
    ConformalLoss, 
    CLIPLoss, 
    TrOCRLoss, 
    FontClassLoss, 
    IFLoss,
    IFLossSinglePass)
from config import set_config
from utils import (
    check_and_create_dir,
    get_data_augs,
    alignment,
    save_image,
    preprocess,
    combine_svgs,
    learning_rate_decay,
    combine_word,
    create_video)
import wandb
import warnings
import torchvision.transforms as tvt
import torchvision
from IF_pipe import IFPipeline
from StyleClassifier import DiscriminatorWithClassifier
import random
from PIL import Image
import torch.nn.functional as F
import string
import kornia.filters as KF
import yaml
warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters


if __name__ == "__main__":

    cfg = set_config()
    svg_path = "/scratch/gilbreth/zhao969/Word-As-Image/merge/the/the"

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    word = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
    # preprocess(cfg.font, word, cfg.optimized_letter, cfg.level_of_cc)

    if cfg.loss.use_if_loss:
        
        prompt_init = f"An image of text \"{cfg.word}\""
        prompt_target = f"An image of text \"{cfg.word}\""
        if_loss = IFLossSinglePass(cfg, device, init_prompt=prompt_init, target_prompt=prompt_target)
    
    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    current_experiment_dir = f"{cfg.experiment_dir}_{cfg.word}"
    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(
        svg_path=svg_path, trainable=cfg.trainable)

    # render init image
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
        torch.ones(img_init.shape[0], img_init.shape[1],
                3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss_target = ToneLoss(cfg)
        tone_loss_target.set_image_init(img_init)

    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            current_experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    #Sweep the learning rate base 
    #pg = [{'params': parameters["point"], 'lr': cfg.sweep_lr_base}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss_target = ConformalLoss(
            parameters, device, "t", shape_groups) # the string should be not used?
    

    def lr_lambda(step): return learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                    lr_delay_steps=cfg.lr.lr_delay_steps,
                                                    lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda,
                        last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        loss_sum = 0.0
        if cfg.use_wandb:
            wandb.log(
                {"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + \
            torch.ones(img.shape[0], img.shape[1], 3,
                    device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        
        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = tvt.functional.invert(data_augs.forward(tvt.functional.invert(x))
                                    )  # perform data aug with black background


        if cfg.use_wandb and cfg.loss.use_font_loss:
            wandb.log({"font_loss": loss.item()}, step=step)

        if cfg.loss.use_if_loss:
            loss_init, loss_target = if_loss(x_aug)
            loss = loss_init * 0.5 + loss_target * 0.5
            if cfg.use_wandb:
                wandb.log({"IF_sds_loss": loss.item()}, step=step)


        if cfg.loss.tone.use_tone_loss:
            tone_target_loss = tone_loss_target(x, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_target_loss}, step=step)
            loss = loss + tone_target_loss

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles_target = conformal_loss_target()
            loss_angles_target = cfg.loss.conformal.angeles_w * loss_angles_target

            loss_angles = loss_angles_target
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()


    filename = os.path.join(
        current_experiment_dir, "output-svg", "output_word.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)
    

    if cfg.use_wandb:
        wandb.finish()
