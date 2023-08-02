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
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video)
import wandb
import warnings
import torchvision.transforms as tvt
import torchvision
from IF_pipe import IFPipeline

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

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    word = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
    preprocess(cfg.font, word, cfg.optimized_letter, cfg.level_of_cc)

    if cfg.loss.use_if_loss:
        if cfg.init_char.islower():
            prompt_pre_init = "An image of the lower case letter "
        else:
            prompt_pre_init = "An image of the upper case letter "

        if cfg.init_style != "none":
            prompt_post_init = " in " + cfg.init_style + " style"
        else:
            prompt_post_init = ""
            
        if cfg.target_char.islower():
            prompt_pre_target = "An image of the lower case letter "
        else:
            prompt_pre_target = "An image of the upper case letter "
        
        if cfg.target_style != "none":
            prompt_post_target = " in " + cfg.target_style + " style"
        else:
            prompt_post_target = ""
        prompt_init = prompt_pre_init + cfg.init_char + prompt_post_init
        prompt_target = prompt_pre_target + cfg.target_char + prompt_post_target
        #pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp32", torch_dtype=torch.float32)
        #del pipe
        if_loss = IFLossSinglePass(cfg, device, init_prompt=prompt_init, target_prompt=prompt_target)

    if cfg.loss.use_font_loss:
        #checkpoint = torch.load('/home/zhao969/Documents/saved_checkpoints/checkpoint_500.pt')
        checkpoint = torch.load('/scratch/gilbreth/zhao969/FontClassifier/saved_checkpoints/case_sensitive_checkpoint_20.pt')
        model_state = checkpoint['model_state_dict']
        font_loss = FontClassLoss(device, model_state, cfg.target_char, 'arial', batch_size=cfg.batch_size, rotate=True)

    if cfg.loss.use_dual_font_loss:
        #checkpoint = torch.load('/home/zhao969/Documents/saved_checkpoints/checkpoint_500.pt')
        checkpoint = torch.load('/scratch/gilbreth/zhao969/FontClassifier/saved_checkpoints/case_sensitive_checkpoint_20.pt')
        model_state = checkpoint['model_state_dict']
        font_loss_rotate = FontClassLoss(device, model_state, cfg.target_char, 'arial', batch_size=cfg.batch_size)
        font_loss_normal = FontClassLoss(device, model_state, cfg.init_char, 'arial',
                                         batch_size=cfg.batch_size, rotate=False)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(
        svg_path=cfg.target, trainable=cfg.trainable)


    # render init image
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
        torch.ones(img_init.shape[0], img_init.shape[1],
                   3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]


    # render target image
    shapes_init, shape_groups_init, parameters_init = init_shapes(svg_path=cfg.init, trainable=cfg.trainable)
    scene_args_init = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
    img_init_init = render(w, h, 2, 2, 0, None, *scene_args_init)
    img_init_init = img_init_init[:, :, 3:4] * img_init_init[:, :, :3] + \
        torch.ones(img_init_init.shape[0], img_init_init.shape[1],
                    3, device=device) * (1 - img_init_init[:, :, 3:4])
    img_init_init = img_init_init[:, :, :3]

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss_target = ToneLoss(cfg)
        tone_loss_init = ToneLoss(cfg)
        tone_loss_target.set_image_init(img_init)
        tone_loss_init.set_image_init(img_init_init)

    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    
    pg = [{'params': parameters["point"] + parameters_init["point"], 'lr': cfg.lr_base["point"]}]
    #Sweep the learning rate base 
    #pg = [{'params': parameters["point"], 'lr': cfg.sweep_lr_base}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss_target = ConformalLoss(
            parameters, device, cfg.optimized_letter, shape_groups)
        conformal_loss_init = ConformalLoss(
            parameters_init, device, cfg.init_char, shape_groups_init)
    

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

        scene_args_init = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
        img_init_init = render(w, h, 2, 2, 0, None, *scene_args_init)
        img_init_init = img_init_init[:, :, 3:4] * img_init_init[:, :, :3] + \
            torch.ones(img_init_init.shape[0], img_init_init.shape[1],
                        3, device=device) * (1 - img_init_init[:, :, 3:4])
        img_init_init = img_init_init[:, :, :3]
        transform = lambda x: torchvision.transforms.functional.rotate(x, 180)

        #img = transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0)
        img = torch.min(img, transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0))

        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, os.path.join(cfg.experiment_dir,
                       "video-png", f"iter{step:04d}.png"), gamma)
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = tvt.functional.invert(data_augs.forward(tvt.functional.invert(x))
                                      )  # perform data aug with black background

        if cfg.loss.use_font_loss:
            loss = font_loss.forward(x_aug) * cfg.loss.font_loss_weight
            #loss = font_loss.forward(x_aug) * cfg.font_loss_weight_sweep

        if cfg.loss.use_dual_font_loss:
            loss = font_loss_rotate.forward(x_aug)
            loss = loss + font_loss_normal.forward(x_aug) * cfg.loss.dual_font_loss_weight

        if cfg.use_wandb and cfg.loss.use_font_loss:
            wandb.log({"font_loss": loss.item()}, step=step)

        if cfg.loss.use_if_loss:
            loss_init, loss_target = if_loss(x_aug)
            #loss = loss_target
            loss = loss_init + loss_target 
            if cfg.use_wandb:
                wandb.log({"IF_sds_loss": loss.item()}, step=step)

        if cfg.loss.tone.use_tone_loss:
            tone_target_loss = tone_loss_target(x, step)
            tone_init_loss = tone_loss_init(x, step) * cfg.loss.dual_bias_weight
            tone_loss_res = tone_target_loss + tone_init_loss
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles_target = conformal_loss_target()
            loss_angles_target = cfg.loss.conformal.angeles_w * loss_angles_target

            loss_angle_init = conformal_loss_init()
            loss_angle_init = cfg.loss.conformal.angeles_w * loss_angle_init

            loss_angles = loss_angles_target + loss_angle_init * cfg.loss.dual_bias_weight
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output_1.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)
    
    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output_2.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes_init, shape_groups_init)

    #combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)

    if cfg.save.image:
        filename = os.path.join(
            cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir,
                     cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
