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
    stack_all_svgs,
    render_svgs_all_type,
    render_from_shapes_and_groups,
    prompt_gen,
    rotate_svg,
    scale_into_wordsize,
    create_video)
import shutil
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
import re
import glob

from main_align import generate_template_svg, generate_all_template_svg
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


def create_word(word, merge_folder, my_font="ambigram_font"):
    if os.path.exists(merge_folder) and os.path.isdir(merge_folder):
        shutil.rmtree(merge_folder)
    
    os.makedirs(merge_folder)

    svgs_conversion = []
    conversion_weight = []
    for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/*.svg"):
        filename = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
        conversion = filename[0:6]
        svgs_conversion.append(conversion)
        conversion_weight.append(float(filename[7:]))

    svgs_lower = [svg.lower() for svg in svgs_conversion]
    for i in range(len(word)):
        if (f"{word[i].lower()}_to_{word[len(word)-1-i].lower()}" not in svgs_lower):
            if (f"{word[len(word)-1-i].lower()}_to_{word[i].lower()}" in svgs_lower):
                index = svgs_lower.index(f"{word[len(word)-1-i].lower()}_to_{word[i].lower()}")
                rotate_svg(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index]}_{conversion_weight[index]}.svg", 
                            f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index][-1]}_to_{svgs_conversion[index][0]}_{1.0-float(conversion_weight[index])}.svg")
            else:
                raise Exception("Cannot find the svg for the character conversion.")
    
    folder_svg = []
    for f in os.listdir(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/"):
        if f.endswith('.svg'):
            folder_svg.append(f)
    folder_svg_lower = ["_".join(svg.lower().split("_")[0:-1]) for svg in folder_svg]
    order_svg = []
    for i in range(len(word)):
        order_svg.append(folder_svg[folder_svg_lower.index(f"{word[i].lower()}_to_{word[len(word)-1-i].lower()}")])
    template_word = "O" * len(word)
    template_word_path = f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{template_word}.svg"
    count = [1] * 26
    template_file = f"{merge_folder}/template.svg"
    shutil.copy(template_word_path, template_file)
    # template_file = generate_template_svg(template_word=template_word, template_name=f"{merge_folder}/template.svg")
    # replicate_shapes(template_file, template_file)
    for i in range(len(word)):
        shutil.copy(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{order_svg[i]}", f"{merge_folder}/{i}_{order_svg[i]}")
    for i in range(len(word)):
        scale_into_wordsize(template_word, template_word[i], template_file, f"{merge_folder}/{i}_{order_svg[i]}", num_rec=count[ord(template_word[i])-ord('A')])
        count[ord(template_word[i])-ord('A')] = count[ord(template_word[i])-ord('A')] + 1
    order_svg_temp = []
    for idx, name in enumerate(order_svg):
        order_svg_temp.append(f"{merge_folder}/{idx}_{name}")
    # order_svg = [f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/{name}" for name in order_svg]
    stack_all_svgs(order_svg_temp, template_file)
    shutil.copy(template_file, f"{merge_folder}/{word}.svg")


if __name__ == "__main__":
    # generate_all_template_svg()

    cfg = set_config()

    words = []
    with open("/scratch/gilbreth/zhao969/BenchMark/words.txt", "r") as input_file:
        for line in input_file:
            word = line.strip()
            words.append(word)
    group_size = len(words) // 20
    groups = [words[i:i + group_size] for i in range(0, len(words), group_size)]
    group_list = groups[cfg.word_group]
    # group_list = ["earth"]
    for opt_word in group_list:
    
        result_svg_folder = f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{opt_word}"
        create_word(opt_word, result_svg_folder, my_font="ambigram_font")
        folder_svg = []
        for f in os.listdir(result_svg_folder):
            if f.endswith('.svg') and f != "template.svg" and f != f"{opt_word}.svg":
                folder_svg.append(f)
        
        order_svg = []
        dual_bias_weight = []
        init_word = ""
        target_word = ""
        for i in range(len(opt_word)):
            for file in glob.glob(f'{result_svg_folder}/{i}_*.svg'):
                order_svg.append(file)
                filename = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
                result = re.search("(.*)_([a-zA-Z])_to_([a-zA-Z])_(.*)", filename)
                dual_bias_weight.append(float(result.group(4)))
                init_word += result.group(2)
                target_word += result.group(3)

        # use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = pydiffvg.get_device()

        prompt_embeds_init = []
        prompt_embeds_target = []
        if cfg.loss.use_if_loss:
            # init_word = opt_word
            # target_word = opt_word
            prompt_init_word = f"An image of the word \"{opt_word}\""
            prompt_target_word = f"An image of the word \"{opt_word}\""
            init_prompt = []
            target_prompt = []
            
            for i in range(len(init_word)):
                prompt_pre_init, prompt_pre_target = prompt_gen(init_word[i], target_word[i])
                init_prompt.append(prompt_pre_init)
                target_prompt.append(prompt_pre_target)

            if_loss = IFLossSinglePass(cfg, device, init_prompt=prompt_init_word, target_prompt=prompt_target_word, precomputed_embeds=True)

            embeds_init, embeds_target = if_loss.prompt_encode(prompt_init_word, prompt_target_word)
            prompt_embeds_init.append(embeds_init)
            prompt_embeds_target.append(embeds_target)

            for i in range(len(init_word)):
                embeds_init, embeds_target = if_loss.prompt_encode(init_prompt[i], target_prompt[i])
                prompt_embeds_init.append(embeds_init)
                prompt_embeds_target.append(embeds_target)
        
        if cfg.loss.use_dual_font_loss or cfg.tuning.font_loss:
            #checkpoint = torch.load('/home/zhao969/Documents/saved_checkpoints/checkpoint_500.pt')
            checkpoint = torch.load('/scratch/gilbreth/zhao969/FontClassifier/saved_checkpoints/case_sensitive_checkpoint_20.pt')
            model_state = checkpoint['model_state_dict']
            font_loss_rotate = FontClassLoss(device, model_state, cfg.target_char, 'Arial', batch_size=cfg.batch_size)
            font_loss_normal = FontClassLoss(device, model_state, cfg.init_char, 'Arial',
                                            batch_size=cfg.batch_size, rotate=False)
            
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
        symmetry_criteria = torch.nn.MSELoss(reduction='sum')

        # initialize shape
        word_img, individual_img_list, shapes_list, shapes_group_list, parameters = render_svgs_all_type(order_svg)

        if cfg.use_wandb:
            plt.imshow(word_img.detach().cpu())
            wandb.log({"init": wandb.Image(plt)}, step=0)
            plt.close()

        num_iter = cfg.num_iter
        
        pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
        optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)


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
            word_img, individual_img_list = render_from_shapes_and_groups(shapes_list, shapes_group_list, step, order_svg)
        
            for idx in range(len(individual_img_list) + 1):
                loss = 0.0
                if idx == 0 and step % 5 == 0:
                    img = word_img
                elif idx != 0:
                    img = individual_img_list[idx - 1]
                else:
                    continue
        
                # if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1) and idx == 0:
                #     save_image(word_img, os.path.join(cfg.experiment_dir,
                #             "video-png", f"iter{step:04d}.png"), gamma)
                    
                #     if cfg.use_wandb:
                #         plt.imshow(word_img.detach().cpu())
                #         wandb.log({"img": wandb.Image(plt)}, step=step)
                #         plt.close()
                
                if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1) and idx == len(individual_img_list):
                    save_image(word_img, os.path.join(cfg.experiment_dir,
                            "video-png", f"iter{step:04d}.png"), gamma)
                    
                    if cfg.use_wandb:
                        plt.imshow(word_img.detach().cpu())
                        wandb.log({"img": wandb.Image(plt)}, step=step)
                        plt.close()

                x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
                x = x.repeat(cfg.batch_size, 1, 1, 1)
                x_aug = tvt.functional.invert(data_augs.forward(tvt.functional.invert(x))
                                            )  # perform data aug with black background


                if cfg.loss.use_if_loss or cfg.loss.use_dreambooth_if:
                    loss_init, loss_target = if_loss(x_aug, prompt_embeds_init[idx], prompt_embeds_target[idx])
                    if idx == 0:
                        loss = loss_init * (0.5) + loss_target * (0.5)
                    else:
                        loss = loss_init * (1 - dual_bias_weight[idx-1]) + loss_target * dual_bias_weight[idx-1]
                    if cfg.use_wandb:
                        wandb.log({"IF_sds_loss": loss.item()}, step=step)
                
                if cfg.loss.use_font_classifier and idx != 0:
                    style_images = random.sample(os.listdir(style_image_path), cfg.batch_size)
                    style_image_tensor = []
                    for image in style_images:
                        style_image_tensor.append(style_transform(Image.open(os.path.join(style_image_path, image)).convert('RGB')))
                    style_image_batch = torch.stack(style_image_tensor)
                    dummy_class = torch.LongTensor(cfg.batch_size * [3]).to(device).to(device)
                    dummy_intensity = torch.LongTensor(cfg.batch_size * [2]).to(device)
                    resize_x_aug = F.interpolate(x_aug, size=64, mode='bilinear', antialias=True)
                    _, style_image_pred, style_gen_pred = discriminator(style_image_batch.to(device), resize_x_aug, dummy_class, dummy_intensity)
                    font_classifier_loss = font_classifier_criterion(style_gen_pred, style_image_pred)
                    loss = loss + font_classifier_loss * (0.3 + 1.0 * torch.exp(-0.005 * torch.tensor(step + 500))) # * torch.exp(-0.005 * torch.tensor(step)) * 2.0 
                    if cfg.use_wandb:
                        wandb.log({"font_classifier_loss": font_classifier_loss.item()}, step=step)
                loss.backward(retain_graph=True)
            
            # Enforce the rotation symmetry
            for idx in range(len(individual_img_list)):
                if (len(individual_img_list) - 1 - idx) == idx:
                    continue # do not enforce symmetry for the middle character
                left_img = individual_img_list[idx].unsqueeze(0).permute(0, 3, 1, 2)
                left_img_blur = KF.median_blur(left_img, (5, 5))
                right_img = individual_img_list[len(individual_img_list) - 1 - idx].unsqueeze(0).permute(0, 3, 1, 2)
                right_img = torchvision.transforms.functional.rotate(right_img, 180)
                right_img_blur = KF.median_blur(right_img, (5, 5))
                loss = 0.0
                # loss = loss + symmetry_criteria(left_img, right_img) / 10000.0
                loss = loss + symmetry_criteria(left_img_blur, right_img_blur) / 10000.0
                if cfg.use_wandb:
                    wandb.log({"symmetry_loss": loss.item()}, step=step)
                loss.backward(retain_graph=True)

            t_range.set_postfix({'loss': loss.item()})
            # loss.backward()
            optim.step()
            scheduler.step()

        word_img, individual_img_list = render_from_shapes_and_groups(shapes_list, shapes_group_list, step, order_svg)
        blured_individual_img_list = []
        for img in individual_img_list:
            blured_individual_img_list.append(KF.median_blur(img.unsqueeze(0).permute(0, 3, 1, 2).detach(), (5, 5)))
        blur_steps = 100
        blur_loss = torch.nn.MSELoss()
        for step in range(blur_steps):
            loss_sum = 0.0
            if cfg.use_wandb:
                wandb.log(
                    {"learning_rate": optim.param_groups[0]['lr']}, step=step + num_iter)
            optim.zero_grad()

            word_img, individual_img_list = render_from_shapes_and_groups(shapes_list, shapes_group_list, step, order_svg)
        
            for idx in range(len(blured_individual_img_list)):
                loss = 0.0
                img = individual_img_list[idx]
                
                if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1) and idx == 0:
                    save_image(word_img, os.path.join(cfg.experiment_dir,
                            "video-png", f"iter{step:04d}.png"), gamma)
                    
                    if cfg.use_wandb:
                        plt.imshow(word_img.detach().cpu())
                        wandb.log({"img": wandb.Image(plt)}, step=step + num_iter)
                        plt.close()
                
                img = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
                blur_loss(img, blured_individual_img_list[idx]).backward()
            optim.step()
            scheduler.step()

        word_img, individual_img_list = render_from_shapes_and_groups(shapes_list, shapes_group_list, step, order_svg)
        word_img_filtered = KF.median_blur(word_img.unsqueeze(0).permute(0, 3, 1, 2).detach(), (5, 5))
        word_img_filtered = word_img_filtered.squeeze().permute(1, 2, 0)
        if cfg.save.image:
            hyper_classification_loss = torch.tensor(0.0)
            filename = os.path.join(
                cfg.experiment_dir, "output-png", "output.png")
            check_and_create_dir(filename)
            imshow = word_img_filtered.detach().cpu()
            pydiffvg.imwrite(imshow, filename, gamma=gamma)
            if cfg.use_wandb:
                plt.imshow(word_img_filtered.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=blur_steps + num_iter + 1)
                plt.close()
        output_svgs = []
        for idx, (shape, shape_group) in enumerate(zip(shapes_list, shapes_group_list)):
            filename = os.path.join(cfg.experiment_dir, "output-svg", f"output_{idx}.svg")
            output_svgs.append(filename)
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shape, shape_group)
            
        stack_all_svgs(output_svgs, os.path.join(cfg.experiment_dir, "output-svg", f"output.svg"))



        import svgutils
        scale = 4
        originalSVG = svgutils.compose.SVG(os.path.join(cfg.experiment_dir, "output-svg", f"output.svg"))
        originalSVG.scale(scale)
        w, h = scale * 224, scale * 224
        device = pydiffvg.get_device()
        figure = svgutils.compose.Figure(float(224) * scale, float(224) * scale, originalSVG)
        figure.save(os.path.join(cfg.experiment_dir, "output-svg", f"scaled_{opt_word}.svg"))
        render = pydiffvg.RenderFunction.apply
        canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
                    os.path.join(cfg.experiment_dir, "output-svg", f"scaled_{opt_word}.svg"))
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                    224 * 4, 224 * 4, shapes_init, shape_groups_init)
                
        word_img = render(w, h, 2, 2, 0, None, *scene_args)
        word_img = word_img[:, :, 3:4] * word_img[:, :, :3] + \
            torch.ones(word_img.shape[0], word_img.shape[1],
                    3, device=device) * (1 - word_img[:, :, 3:4])
        word_img = word_img[:, :, :3]
        x = word_img.unsqueeze(0).permute(0, 3, 1, 2)
        x_blured = KF.median_blur(x.detach(), (5, 5))
        x_blured = KF.median_blur(x_blured.detach(), (5, 5))
        check_and_create_dir(os.path.join(cfg.experiment_dir, "output-png", f"scaled_{opt_word}.png"))
        plt.imsave(os.path.join(cfg.experiment_dir, "output-png", f"scaled_{opt_word}.png"), x_blured.squeeze().permute(1, 2, 0).detach().cpu().numpy())

    if cfg.use_wandb:
        wandb.finish()
    