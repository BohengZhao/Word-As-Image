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
    rare_tokens = ["nips" ,"rune" ,"pafc" ,"nlwx" ,"fck", "cae", "sohn" ,"dori"
                ,"zawa", "revs", "hmv", "whoo", "asar", "blob", "emp", "kilt", 
                "ulf", "loeb", "bnha", "gote", "omd", "adl", "shld" , "waj", "sown" ,"rcn"]
    # weight_sweep_list = [0.4, 0.45, 0.5, 0.55, 0.6]
    weight_sweep_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    weight_sweep_list = [0.35, 0.38, 0.42, 0.45]
    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    # with open("/scratch/gilbreth/zhao969/Word-As-Image/code/config/style.yaml", 'r') as f:
    #     style_info = yaml.load(f, Loader=yaml.FullLoader)
    # style_info = torch.Tensor(list(style_info.values()))
    # values = [random.uniform(0, 1) for i in range(37)]
    #values = [0.5] * 37
    #values[1] = 1.0
    #values[34] = 1.0
    #style_info = torch.Tensor(values)
    # values = [56.23,22.21,7.63,11.74,62.39,96.57,76.27,0,28.6,25.09,19.8,4.3,38.92,44.91,10,39.7,48.53,66.99,44.83,81.17,22.74,3.07,9.37,41.4,0,100,16.9,11.91,9.96,60.81,28.59,86.31,7.57,9.94,94.47,27.76,56.95]
    # values = [26.48,32.19,53.55,62.69,13.32,41.52,75.72,0,64.08,26.8,28.06,4.3,41.28,13.43,7.46,38.43,73.39,91.8,80.55,89.3,61.98,47.21,9.34,89.24,42.18,0,36.56,42.21,9.43,59.17,20.8,64.95,64.78,34.69,34.05,76.23,45.18]
    # values = [value / 100.0 for value in values]
    # style_info = torch.Tensor(values)
    # style_info = style_info.unsqueeze(0).repeat(cfg.batch_size, 1)

    print("preprocessing")
    word = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
    # preprocess(cfg.font, word, cfg.optimized_letter, cfg.level_of_cc)

    if cfg.loss.use_if_loss:
        if cfg.init_char.islower():
            prompt_pre_init = "An image of the lower case letter "
        else:
            prompt_pre_init = "An image of the upper case letter "
            
        if cfg.target_char.islower():
            prompt_pre_target = "An image of the lower case letter "
        else:
            prompt_pre_target = "An image of the upper case letter "
        
        prompt_init = prompt_pre_init + cfg.init_char #+ "in curly font style"
        prompt_target = prompt_pre_target + cfg.target_char #+ "in curly font style"
        prompt_init = "An image of the text \'ls\'"
        prompt_target = "An image of the upper case letter R"
        if_loss = IFLossSinglePass(cfg, device, init_prompt=prompt_init, target_prompt=prompt_target)

    if cfg.loss.use_dual_font_loss or cfg.tuning.font_loss:
        checkpoint = torch.load('/home/zhao969/Documents/saved_checkpoints/checkpoint_500.pt')
        #checkpoint = torch.load('/scratch/gilbreth/zhao969/FontClassifier/saved_checkpoints/case_sensitive_checkpoint_20.pt')
        model_state = checkpoint['model_state_dict']
        font_loss_rotate = FontClassLoss(device, model_state, cfg.target_char, 'Arial', batch_size=cfg.batch_size, case_sensitive=False)
        font_loss_normal = FontClassLoss(device, model_state, cfg.init_char, 'Arial',
                                         batch_size=cfg.batch_size, rotate=False, case_sensitive=False)
        
    if cfg.loss.use_font_classifier:
        # style_image_path = "/scratch/gilbreth/zhao969/Word-As-Image/code/data/selected_font_img/Quarterly"
        style_image_path = "/scratch/gilbreth/zhao969/Word-As-Image/code/data/selected_font_img/Typewriter"
        import torchvision.transforms as T
        load_epoch = 500
        checkpoint_dir = "/scratch/gilbreth/zhao969/Attr2Font/experiments/att2font_en/checkpoint"
        discriminator = DiscriminatorWithClassifier()
        discriminator.eval()
        dis_file = os.path.join(checkpoint_dir, f"D_{load_epoch}.pth")
        discriminator.load_state_dict(torch.load(dis_file))
        discriminator = discriminator.to(device)
        font_classifier_criterion = torch.nn.MSELoss()
        style_transform = []
        style_transform.append(T.Resize(64))
        style_transform.append(T.ToTensor())
        style_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        style_transform = T.Compose(style_transform)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    for sweep_weight in weight_sweep_list:
        current_experiment_dir = f"{cfg.experiment_dir}_{sweep_weight}"
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
        align_mode = "seperated"
        # shift_x = alignment(img_init, img_init_init, mode=cfg.alignment)
        shift_x = alignment(img_init, img_init_init, mode=align_mode, rotate=False)

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
                current_experiment_dir, "svg-init", "init.svg")
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
            # transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
            transform = lambda x: x
            #img = transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0)
            # img = torch.min(img, transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0))
            #shift_x = alignment(img, img_init_init, mode='overlap')
            #img = torch.min(torch.roll(img, -(shift_x // 2), 1), torch.roll(transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0), shift_x // 2, 1))
            if shift_x != 0:
                img = torch.min(torch.roll(img, -(shift_x // 2), 1), torch.roll(transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0), shift_x // 2, 1))
            else:
                img = torch.min(img, transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0))
            if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
                # save_image(img, os.path.join(cfg.experiment_dir,
                #            "video-png", f"iter{step:04d}.png"), gamma)
                # filename = os.path.join(
                #     cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
                # check_and_create_dir(filename)
                # save_svg.save_svg(
                #     filename, w, h, shapes, shape_groups)
                if cfg.use_wandb:
                    plt.imshow(img.detach().cpu())
                    wandb.log({"img": wandb.Image(plt)}, step=step)
                    plt.close()

            x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
            x = x.repeat(cfg.batch_size, 1, 1, 1)
            x_style = x
            x_aug = tvt.functional.invert(data_augs.forward(tvt.functional.invert(x))
                                        )  # perform data aug with black background

            # if cfg.use_wandb and cfg.loss.use_font_loss:
            #     wandb.log({"font_loss": loss.item()}, step=step)

            if cfg.loss.use_if_loss or cfg.loss.use_dreambooth_if:
                loss_init, loss_target = if_loss(x_aug)
                #loss = loss_target
                loss = loss_init * (1 - sweep_weight) + loss_target * sweep_weight
                if cfg.use_wandb:
                    wandb.log({"IF_sds_loss": loss.item()}, step=step)

            if cfg.loss.use_font_loss:
                loss = loss +  font_loss.forward(x_aug) * cfg.loss.font_loss_weight
                #loss = font_loss.forward(x_aug) * cfg.font_loss_weight_sweep

            if cfg.loss.use_dual_font_loss:
                classification_loss = font_loss_rotate.forward(x_aug)[0] + font_loss_normal.forward(x_aug)[0] #* cfg.loss.dual_font_loss_weight
                classification_loss = classification_loss * torch.exp(-0.5 * torch.tensor(step)) * 0.001
                loss = loss + classification_loss 

                font_loss = font_loss_rotate.forward(x_aug)[1] + font_loss_normal.forward(x_aug)[1]
                font_loss = font_loss * torch.exp(-0.1 * torch.tensor(step))
                loss = loss - font_loss 
                if cfg.use_wandb:
                    wandb.log({"classification_loss": classification_loss.item()}, step=step)
                    wandb.log({"font_loss": font_loss.item()}, step=step)

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

            if cfg.loss.use_font_classifier:
                style_images = random.sample(os.listdir(style_image_path), cfg.batch_size)
                style_image_tensor = []
                for image in style_images:
                    style_image_tensor.append(style_transform(Image.open(os.path.join(style_image_path, image)).convert('RGB')))
                style_image_batch = torch.stack(style_image_tensor)
                dummy_class = torch.LongTensor(cfg.batch_size * [3]).to(device).to(device)
                dummy_intensity = torch.LongTensor(cfg.batch_size * [2]).to(device)
                resize_x_aug = F.interpolate(x_style, size=64, mode='bilinear', antialias=True)
                _, style_image_pred, style_gen_pred = discriminator(style_image_batch.to(device), resize_x_aug, dummy_class, dummy_intensity)
                # print(style_image_pred[0])
                # font_classifier_loss = font_classifier_criterion(style_gen_pred, style_info.to(device))
                font_classifier_loss = font_classifier_criterion(style_gen_pred, style_image_pred)
                loss = loss + font_classifier_loss * (0.5 + 3.0 * torch.exp(-0.005 * torch.tensor(step))) # * torch.exp(-0.005 * torch.tensor(step)) * 2.0 
                # loss = loss + font_classifier_loss * (0.5 + 2.5 * torch.exp(-0.005 * torch.tensor(step)))
                if cfg.use_wandb:
                    wandb.log({"font_classifier_loss": font_classifier_loss.item()}, step=step)

            t_range.set_postfix({'loss': loss.item()})
            loss.backward()
            optim.step()
            scheduler.step()


        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)
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
        # transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
        transform = lambda x: x

        if shift_x != 0:
            img = torch.min(torch.roll(img, -(shift_x // 2), 1), torch.roll(transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0), shift_x // 2, 1))
        else:
            img = torch.min(img, transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0))
        # plt.imsave("dual.png", img.detach().cpu().numpy())
        x = img.unsqueeze(0).permute(0, 3, 1, 2)
        x_blured = KF.median_blur(x.detach(), (5, 5))
        # plt.imsave("dual_blured_5.png", x_blured.squeeze().permute(1, 2, 0).detach().cpu().numpy())

        if cfg.save.image:
            # hyper_classification_loss = torch.tensor(0.0)
            # if cfg.tuning.font_loss:
            #     hyper_classification_rotate_loss = font_loss_rotate.forward(x_aug)[0]
            #     hyper_classification_normal_loss = font_loss_normal.forward(x_aug)[0]
            #     hyper_classification_loss = hyper_classification_rotate_loss + hyper_classification_normal_loss
            filename = os.path.join(
                current_experiment_dir, "output-png", f"output.png")
            check_and_create_dir(filename)
            imshow = img.detach().cpu()
            pydiffvg.imwrite(imshow, filename, gamma=gamma)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        # if cfg.save.video:
        #     print("saving video")
        #     create_video(cfg.num_iter, cfg.experiment_dir,
        #                  cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
