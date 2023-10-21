import collections.abc
import os
import os.path as osp
from torch import nn
import kornia.augmentation as K
import pydiffvg
import save_svg
import cv2
from ttf import font_string_to_svgs, normalize_letter_size
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import svgutils
import svgutils.transform as sg
from easydict import EasyDict as edict
import copy
import math

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def preprocess(font, word, letter, level_of_cc=1):

    if level_of_cc == 0:
        target_cp = None
    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }
        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    print(f"======= {font} =======")
    font_path = f"code/data/fonts/{font}.ttf"
    init_path = f"code/data/init"
    subdivision_thresh = None
    font_string_to_svgs(init_path, font_path, word, target_control=target_cp,
                        subdivision_thresh=subdivision_thresh)
    normalize_letter_size(init_path, font_path, word)

    # optimaize two adjacent letters
    if len(letter) > 1:
        subdivision_thresh = None
        font_string_to_svgs(init_path, font_path, letter, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(init_path, font_path, letter)

    print("Done preprocess")


def get_data_augs(cut_size):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(K.RandomCrop(size=(cut_size, cut_size), pad_if_needed=True, padding_mode='reflect', p=1.0))
    return nn.Sequential(*augmentations)


'''pytorch adaptation of https://github.com/google/mipnerf'''
def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp



def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)


def get_letter_ids(letter, word, shape_groups):
    for group, l in zip(shape_groups, word):
        if l == letter:
            return group.shape_ids


def combine_word(word, letter, font, experiment_dir):
    def get_letter_ids_for_overlayed(letter, word, shape_groups):
        for group, l in zip(shape_groups, word * 2):
            if l == letter:
                return group.shape_ids
    word_svg_scaled = f"./code/data/init/{font}_{word}_scaled.svg"
    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
    letter_ids = []
    for l in letter:
        letter_ids += get_letter_ids_for_overlayed(l, word, shape_groups_word)

    w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2

    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes])
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes])

    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    if scale_canvas_h > scale_canvas_w:
        wsize = int((out_w_max - out_w_min) * scale_canvas_h)
        scale_canvas_w = wsize / (out_w_max - out_w_min)
        shift_w = -out_c_w * scale_canvas_w + c_w
    else:
        hsize = int((out_h_max - out_h_min) * scale_canvas_w)
        scale_canvas_h = hsize / (out_h_max - out_h_min)
        shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        if scale_canvas_h > scale_canvas_w:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
        else:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

    for j, s in enumerate(letter_ids):
        shapes_word[s] = shapes[j]

    save_svg.save_svg(
        f"{experiment_dir}/{font}_{word}_{letter}.svg", canvas_width, canvas_height, shapes_word,
        shape_groups_word)

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes_word, shape_groups_word)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
               torch.ones(img.shape[0], img.shape[1], 3, device="cuda:0") * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    save_image(img, f"{experiment_dir}/{font}_{word}_{letter}.png")


def create_video(num_iter, experiment_dir, video_frame_freq):
    img_array = []
    for ii in range(0, num_iter):
        if ii % video_frame_freq == 0 or ii == num_iter - 1:
            filename = os.path.join(
                experiment_dir, "video-png", f"iter{ii:04d}.png")
            img = cv2.imread(filename)
            img_array.append(img)

    video_name = os.path.join(
        experiment_dir, "video.mp4")
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (500, 500))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()

# New functions

def diff_norm(input, mean, std):
    if (len(input.shape) == 3):
        for i in range(input.shape[0]):
            input[i] = (input[i] - mean[i]) / std[i]
    elif (len(input.shape) == 4):
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                input[i, j] = (input[i, j] - mean[j]) / std[j]
    return input

def augment(image):
    image = transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.9, 1.1), fill=0)(image)
    return image

def alignment(mask1, mask2, mode='overlap', rotate=True):
    if rotate:
        transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
    else:
        transform = lambda x: x
    if mode == 'overlap':
        highest = 0
        shift = 0
        mask2 = transform(mask2.permute(2, 0, 1)).permute(1, 2, 0)
        for i in range(-100, 100, 1):
            new_mask2 = torch.roll(mask2, i, 1)
            overlap_sum = torch.einsum('ijk,ijk->ij', mask1, new_mask2).unsqueeze(2).sum()
            if overlap_sum > highest:
                highest = overlap_sum
                shift = i
        return shift
    elif mode == 'seperated':
        lowest = float('inf')
        shift = float('inf')
        mask2 = transform(mask2.permute(2, 0, 1)).permute(1, 2, 0)
        for i in range(-100, 0, 1):
            new_mask2 = torch.roll(mask2, i, 1)
            overlap_sum = torch.einsum('ijk,ijk->ij', mask1, new_mask2).unsqueeze(2).sum()
            if overlap_sum <= lowest and abs(i) < abs(shift):
                lowest = overlap_sum
                shift = i
        return math.ceil(shift * 0.7)
    elif mode == 'contact':
        lowest = float('inf')
        shift = float('inf')
        mask2 = transform(mask2.permute(2, 0, 1)).permute(1, 2, 0)
        for i in range(-100, 100, 1):
            new_mask2 = torch.roll(mask2, i, 1)
            overlap_sum = torch.einsum('ijk,ijk->ij', mask1, new_mask2).unsqueeze(2).sum()
            if overlap_sum <= lowest and abs(i) < abs(shift):
                lowest = overlap_sum
                shift = i
        shift = shift // 2
        return shift
    elif mode == 'overlap_2':
        highest = 0
        shift = 0
        mask2 = transform(mask2.permute(2, 0, 1)).permute(1, 2, 0)
        for i in range(-100, 100, 1):
            new_mask2 = torch.roll(mask2, i, 1)
            overlap_sum = torch.einsum('ijk,ijk->ij', mask1, new_mask2).unsqueeze(2).sum()
            if overlap_sum > highest:
                highest = overlap_sum
                shift = i
        return shift * 2
    elif mode == 'seperated_2':
        lowest = float('inf')
        shift = float('inf')
        transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
        mask2 = transform(mask2.permute(2, 0, 1)).permute(1, 2, 0)
        for i in range(100, 0, -1):
            new_mask2 = torch.roll(mask2, i, 1)
            overlap_sum = torch.einsum('ijk,ijk->ij', mask1, new_mask2).unsqueeze(2).sum()
            if overlap_sum <= lowest and abs(i) < abs(shift):
                lowest = overlap_sum
                shift = i
        return math.ceil(shift * 0.7)
    else:
        return 0
    
def combine_svgs(file_1, file_2, shift_x, w=224, h=224):
    svg = svgutils.transform.fromfile(file_2)
    originalSVG = svgutils.compose.SVG(file_2)
    originalSVG.rotate(180, 112, 112)
    originalSVG.move(shift_x // 2, 0)
    figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
    figure.save(file_2)

    svg = svgutils.transform.fromfile(file_1)
    originalSVG = svgutils.compose.SVG(file_1)
    originalSVG.move(-(shift_x // 2), 0)
    figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
    figure.save(file_1)

    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        file_1)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg(file_1, w, h, shapes_init, shape_groups_init)

    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        file_2)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg(file_2, w, h, shapes_init, shape_groups_init)

    
    #create new SVG figure
    fig = sg.SVGFigure(h, w)
    # load matpotlib-generated figures
    fig1 = sg.fromfile(file_1)
    fig2 = sg.fromfile(file_2)
    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    # add text labels
    # append plots and labels to figure
    fig.append([plot1, plot2])
    # save generated SVG files
    fig.save(os.path.join(os.path.dirname(file_1), "output.svg"))
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        os.path.join(os.path.dirname(file_1), "output.svg"))
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg(os.path.join(os.path.dirname(file_1), "output.svg"), w, h, shapes_init, shape_groups_init)

def stack_all_svgs(files, outfile):
    h, w = 224, 224
    fig = sg.SVGFigure(h, w)
    # load matpotlib-generated figures
    fig_list = []
    for file in files:
        fig_list.append(sg.fromfile(file))
    # get the plot objects'
    plot_list = []
    for fig in fig_list:
        plot_list.append(fig.getroot())
    # add text labels
    # append plots and labels to figure
    fig.append(plot_list)
    # save generated SVG files
    fig.save(outfile)
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        outfile)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg(outfile, w, h, shapes_init, shape_groups_init)

def replicate_shapes(svg_file, output_file):
    import svgutils.transform as sg
    h, w = 224, 224
    #create new SVG figure
    fig = sg.SVGFigure(h, w)
    # load matpotlib-generated figures
    fig1 = sg.fromfile(svg_file)
    fig2 = sg.fromfile(svg_file)
    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    # add text labels
    # append plots and labels to figure
    fig.append([plot1, plot2])
    # save generated SVG files
    fig.save(output_file)
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        output_file)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg(output_file, w, h, shapes_init, shape_groups_init)

# def scale_into_wordsize(word, letter, word_svg, letter_svg, num_rec=1):
#     def get_letter_ids_with_recur(letter, word, shape_groups, num_rec=1):
#         count = 1
#         for group, l in zip(shape_groups, word):
#             if (l == letter) and (count == num_rec):
#                 return group.shape_ids
#             elif l == letter:
#                 count += 1
        
#     word_svg_scaled = word_svg
#     canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
#     letter_ids = []
#     for l in letter:
#         letter_ids += get_letter_ids_with_recur(l, word, shape_groups_word, num_rec)

#     w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
#         [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
#     h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
#         [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

#     c_w = (-w_min + w_max) / 2
#     c_h = (-h_min + h_max) / 2

#     svg_result = letter_svg
#     canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

#     out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
#         [torch.max(p.points[:, 0]) for p in shapes])
#     out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
#         [torch.max(p.points[:, 1]) for p in shapes])

#     out_c_w = (-out_w_min + out_w_max) / 2
#     out_c_h = (-out_h_min + out_h_max) / 2

#     scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
#     scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

#     if scale_canvas_h > scale_canvas_w:
#         wsize = int((out_w_max - out_w_min) * scale_canvas_h)
#         scale_canvas_w = wsize / (out_w_max - out_w_min)
#         shift_w = -out_c_w * scale_canvas_w + c_w
#     else:
#         hsize = int((out_h_max - out_h_min) * scale_canvas_w)
#         scale_canvas_h = hsize / (out_h_max - out_h_min)
#         shift_h = -out_c_h * scale_canvas_h + c_h

#     for num, p in enumerate(shapes):
#         p.points[:, 0] = p.points[:, 0] * scale_canvas_w
#         p.points[:, 1] = p.points[:, 1] * scale_canvas_h
#         if scale_canvas_h > scale_canvas_w:
#             p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
#             p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
#         else:
#             p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
#             p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

#     save_svg.save_svg(
#         letter_svg, canvas_width, canvas_height, shapes,
#         shape_groups)

def scale_into_wordsize(word, letter, word_svg, letter_svg, num_rec=1):
    def get_letter_ids_with_recur(letter, word, shape_groups, num_rec=1):
        count = 1
        for group, l in zip(shape_groups, word):
            if (l == letter) and (count == num_rec):
                return group.shape_ids
            elif l == letter:
                count += 1
        
    word_svg_scaled = word_svg
    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
    letter_ids = []
    for l in letter:
        letter_ids += get_letter_ids_with_recur(l, word, shape_groups_word, num_rec)

    w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2

    svg_result = letter_svg
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes])
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes])

    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    wsize = int((out_w_max - out_w_min) * scale_canvas_h)
    scale_canvas_w = wsize / (out_w_max - out_w_min)
    shift_w = -out_c_w * scale_canvas_w + c_w
    hsize = int((out_h_max - out_h_min) * scale_canvas_w)
    scale_canvas_h = hsize / (out_h_max - out_h_min)
    shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
        p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h
    save_svg.save_svg(
        letter_svg, canvas_width, canvas_height, shapes,
        shape_groups)

def prompt_gen(init_letter, target_letter):
    if init_letter.islower():
        prompt_pre_init = "An image of the lower case letter " + init_letter
    else:
        prompt_pre_init = "An image of the upper case letter " + init_letter
            
    if target_letter.islower():
        prompt_pre_target = "An image of the lower case letter " + target_letter
    else:
        prompt_pre_target = "An image of the upper case letter " + target_letter
    return prompt_pre_init, prompt_pre_target
        

def rotate_svg(input_file, output_file, w=224, h=224):
    svg = svgutils.transform.fromfile(input_file)
    originalSVG = svgutils.compose.SVG(input_file)
    originalSVG.rotate(180, w // 2, h // 2)
    figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
    figure.save(output_file)

    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        output_file)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, shapes_init, shape_groups_init)
    save_svg.save_svg(output_file, w, h, shapes_init, shape_groups_init)

def scale_and_save_svg(word, word_svg, letter_svgs, result_svg, num_rec=1):
    def get_letter_ids_with_recur(letter, word, shape_groups, num_rec=1):
        count = 1
        for group, l in zip(shape_groups, word):
            if (l == letter) and (count == num_rec):
                return group.shape_ids
            elif l == letter:
                count += 1
    
    word_svg_scaled = word_svg
    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
    for idx in range(len(word)):
        letter_ids = []
        letter_ids += shape_groups_word[idx].shape_ids

        w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
            [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
        h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
            [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

        c_w = (-w_min + w_max) / 2
        c_h = (-h_min + h_max) / 2

        svg_result = letter_svgs[idx]
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

        out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
            [torch.max(p.points[:, 0]) for p in shapes])
        out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
            [torch.max(p.points[:, 1]) for p in shapes])

        out_c_w = (-out_w_min + out_w_max) / 2
        out_c_h = (-out_h_min + out_h_max) / 2

        scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
        scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

        if scale_canvas_h > scale_canvas_w:
            wsize = int((out_w_max - out_w_min) * scale_canvas_h)
            scale_canvas_w = wsize / (out_w_max - out_w_min)
            shift_w = -out_c_w * scale_canvas_w + c_w
        else:
            hsize = int((out_h_max - out_h_min) * scale_canvas_w)
            scale_canvas_h = hsize / (out_h_max - out_h_min)
            shift_h = -out_c_h * scale_canvas_h + c_h

        for num, p in enumerate(shapes):
            p.points[:, 0] = p.points[:, 0] * scale_canvas_w
            p.points[:, 1] = p.points[:, 1] * scale_canvas_h
            if scale_canvas_h > scale_canvas_w:
                p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
                p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
            else:
                p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
                p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

        save_svg.save_svg(
            svg_result, canvas_width, canvas_height, shapes,
            shape_groups)
        
def render_svgs_all_type(svg_file_list):
    h, w = 224, 224
    target_h_letter = 180
    target_canvas_width, target_canvas_height = 224, 224
    device = pydiffvg.get_device()
    parameters = edict()
    parameters.point = []
    shapes_list = []
    shapes_group_list = []

    render = pydiffvg.RenderFunction.apply
    word_img = None
    individual_img_list = []
    for svg_file in svg_file_list:
        canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
            svg_file)
        shapes_list.append(shapes_init)
        shapes_group_list.append(shape_groups_init)
        
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

        # render init image
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, shapes_init, shape_groups_init)
        if word_img == None:
            word_img = render(w, h, 2, 2, 0, None, *scene_args)
            word_img = word_img[:, :, 3:4] * word_img[:, :, :3] + \
                torch.ones(word_img.shape[0], word_img.shape[1],
                        3, device=device) * (1 - word_img[:, :, 3:4])
            word_img = word_img[:, :, :3]
        else:
            temp = render(w, h, 2, 2, 0, None, *scene_args)
            temp = temp[:, :, 3:4] * temp[:, :, :3] + \
                torch.ones(temp.shape[0], temp.shape[1],
                        3, device=device) * (1 - temp[:, :, 3:4])
            temp = temp[:, :, :3]
            word_img = torch.min(word_img, temp)

        #render individual scaled image
        w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes_init]), max(
            [torch.max(p.points[:, 0]) for p in shapes_init])
        h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes_init]), max(
            [torch.max(p.points[:, 1]) for p in shapes_init])

        letter_h = h_max - h_min
        letter_w = w_max - w_min
        
        scale_canvas_h = target_h_letter / letter_h
        wsize = int(letter_w * scale_canvas_h)
        scale_canvas_w = wsize / letter_w

        _, _, shapes_init_copy, _ = pydiffvg.svg_to_scene(
            svg_file)

        for num, (p, p_copy) in enumerate(zip(shapes_init, shapes_init_copy)):
            p_copy.points[:, 0] = p.points[:, 0] * scale_canvas_w
            p_copy.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter

        w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes_init_copy]), max(
            [torch.max(p.points[:, 0]) for p in shapes_init_copy])
        h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes_init_copy]), max(
            [torch.max(p.points[:, 1]) for p in shapes_init_copy])
        
        _, _, shapes_init_copy_copy, _ = pydiffvg.svg_to_scene(
            svg_file)

        for num, (p, p_copy) in enumerate(zip(shapes_init_copy, shapes_init_copy_copy)):
            p_copy.points[:, 0] = p.points[:, 0] + target_canvas_width / \
                2 - int(w_min + (w_max - w_min) / 2)
            p_copy.points[:, 1] = p.points[:, 1] + target_canvas_height / \
                2 - int(h_min + (h_max - h_min) / 2)
            
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_init_copy_copy, shape_groups_init)
        ind_img = render(w, h, 2, 2, 0, None, *scene_args)

        # compose image with white background
        ind_img = ind_img[:, :, 3:4] * ind_img[:, :, :3] + \
            torch.ones(ind_img.shape[0], ind_img.shape[1], 3,
                        device=device) * (1 - ind_img[:, :, 3:4])
        ind_img = ind_img[:, :, :3]
        individual_img_list.append(ind_img)

        
    return word_img, individual_img_list, shapes_list, shapes_group_list, parameters

def render_from_shapes_and_groups(shapes_list, shapes_group_list, step, svg_file_list):
    device = pydiffvg.get_device()
    render = pydiffvg.RenderFunction.apply
    word_img = None
    individual_img_list = []
    h, w = 224, 224
    target_h_letter = 180
    target_canvas_width, target_canvas_height = 224, 224
    for i in range(len(shapes_list)):
        shapes_init = shapes_list[i]
        shape_groups_init = shapes_group_list[i]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_init, shape_groups_init)
        if word_img == None:
            word_img = render(w, h, 2, 2, step, None, *scene_args)
            word_img = word_img[:, :, 3:4] * word_img[:, :, :3] + \
                torch.ones(word_img.shape[0], word_img.shape[1],
                        3, device=device) * (1 - word_img[:, :, 3:4])
            word_img = word_img[:, :, :3]
        else:
            temp = render(w, h, 2, 2, step, None, *scene_args)
            temp = temp[:, :, 3:4] * temp[:, :, :3] + \
                torch.ones(temp.shape[0], temp.shape[1],
                        3, device=device) * (1 - temp[:, :, 3:4])
            temp = temp[:, :, :3]
            word_img = torch.min(word_img, temp)

        #render individual scaled image
        w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes_init]), max(
            [torch.max(p.points[:, 0]) for p in shapes_init])
        h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes_init]), max(
            [torch.max(p.points[:, 1]) for p in shapes_init])

        letter_h = h_max - h_min
        letter_w = w_max - w_min
        
        scale_canvas_h = target_h_letter / letter_h
        wsize = int(letter_w * scale_canvas_h)
        scale_canvas_w = wsize / letter_w

        #shapes_init_copy = copy.deepcopy(shapes_init)
        # shapes_init_copy = [copy.deepcopy(p) for p in shapes_init]
        _, _, shapes_init_copy, _ = pydiffvg.svg_to_scene(
            svg_file_list[i])

        for num, (p, p_copy) in enumerate(zip(shapes_init, shapes_init_copy)):
            p_copy.points[:, 0] = p.points[:, 0] * scale_canvas_w
            p_copy.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter

        w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes_init_copy]), max(
            [torch.max(p.points[:, 0]) for p in shapes_init_copy])
        h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes_init_copy]), max(
            [torch.max(p.points[:, 1]) for p in shapes_init_copy])
        
        _, _, shapes_init_copy_copy, _ = pydiffvg.svg_to_scene(
            svg_file_list[i])

        for num, (p, p_copy) in enumerate(zip(shapes_init_copy, shapes_init_copy_copy)):
            p_copy.points[:, 0] = p.points[:, 0] + target_canvas_width / \
                2 - int(w_min + (w_max - w_min) / 2)
            p_copy.points[:, 1] = p.points[:, 1] + target_canvas_height / \
                2 - int(h_min + (h_max - h_min) / 2)
            
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_init_copy_copy, shape_groups_init)
        ind_img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        ind_img = ind_img[:, :, 3:4] * ind_img[:, :, :3] + \
            torch.ones(ind_img.shape[0], ind_img.shape[1], 3,
                        device=device) * (1 - ind_img[:, :, 3:4])
        ind_img = ind_img[:, :, :3]
        individual_img_list.append(ind_img)
    return word_img, individual_img_list

def gen_bash_script(init_word, target_word, weight_sweep=[0.4, 0.5, 0.6], alignment=["none", "overlap", "seperated_2", "seperated"], font="IndieFlower-Regular"):
    assert len(init_word) == len(target_word)
    dir = '/scratch/gilbreth/zhao969/Word-As-Image/scripts/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for idx in range(math.ceil(len(init_word) / 2)):
        for alignment_idx in range(len(alignment)):
            init_char = [init_word[idx].upper(), init_word[idx].lower()]
            target_char = [target_word[len(target_word) - 1 - idx].upper(), target_word[len(target_word) - 1 - idx].lower()]
            alignments = alignment[alignment_idx]
            weights = " ".join([str(w) for w in weight_sweep])
            with open (f'/scratch/gilbreth/zhao969/Word-As-Image/scripts/run_{idx * len(alignment) + alignment_idx}.sh', 'w') as bashfile:
                bashfile.write(f'''\
#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

set -e
USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"
EXPERIMENT=dual_if

TARGETS=(\"{target_char[0]}\" \"{target_char[1]}\")
letter_=(\"{init_char[0]}\" \"{init_char[1]}\")
fonts=({font})
alignment=({alignments})
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
for j in \"${{fonts[@]}}\"
do
    SEED=0
    for i in "${{letter_[@]}}"
    do
        for target in "${{TARGETS[@]}}"
        do 
            for align in "${{alignment[@]}}"
            do
                for weight in {weights}
                do
                    echo "$i"
                    font_name=$j
                    ARGS="--experiment $EXPERIMENT --optimized_letter ${{i}} --seed $SEED --font ${{font_name}} --use_wandb ${{USE_WANDB}} --wandb_user ${{WANDB_USER}} --init_char ${{i}} --target_char ${{target}}"
                    CUDA_VISIBLE_DEVICES=0 python code/main_dual.py $ARGS --dual_bias_weight_sweep "${{weight}}" --alignment "${{align}}" --word \"{init_word}\"
                done
            done
        done
    done
done
            ''')


def gen_bash_all(weight_sweep=[0.4, 0.5, 0.6], alignment=["none", "overlap", "seperated_2", "seperated"], font="IndieFlower-Regular"):
    dir = '/scratch/gilbreth/zhao969/Word-As-Image/scripts/'
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for idx in range(len(alphabet)):
        for idx_2 in range(idx, len(alphabet)):
            for align_idx in range(len(alignment)):
                init_char = [alphabet[idx].upper(), alphabet[idx].lower()]
                target_char = [alphabet[idx_2].upper(), alphabet[idx_2].lower()]
                alignments = alignment[align_idx]
                weights = " ".join([str(w) for w in weight_sweep])
                with open (f'/scratch/gilbreth/zhao969/Word-As-Image/scripts/run_{idx}_to_{idx_2}_{align_idx}.sh', 'w') as bashfile:
                    bashfile.write(f'''\
#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

set -e
USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"
EXPERIMENT=dual_if

TARGETS=(\"{target_char[1]}\" \"{target_char[1]}\" \"{target_char[0]}\" \"{target_char[0]}\")
letter_=(\"{init_char[0]}\" \"{init_char[1]}\" \"{init_char[0]}\" \"{init_char[1]}\")
fonts=({font})
alignment=({alignments})
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
length=${{#TARGETS[@]}}

for j in \"${{fonts[@]}}\"
do
    SEED=0
    for (( k=0; k<${{length}}; k++ ));
    do
    target="${{TARGETS[$k]}}"
    i="${{letter_[$k]}}"
        for align in "${{alignment[@]}}"
        do
            font_name=$j
            ARGS="--experiment $EXPERIMENT --optimized_letter ${{i}} --seed $SEED --font ${{font_name}} --use_wandb ${{USE_WANDB}} --wandb_user ${{WANDB_USER}} --init_char ${{i}} --target_char ${{target}}"
            CUDA_VISIBLE_DEVICES=0 python code/main_dual.py $ARGS --dual_bias_weight_sweep "0.4" --alignment "${{align}}" --word \"{alphabet}\"
        done
    done
done
            ''')
        