from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss, CLIPLoss, TrOCRLoss, FontClassLoss, IFLoss
from config import set_config
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    alignment,
    learning_rate_decay,
    combine_word,
    create_video)
import wandb
import warnings
import torchvision.transforms as tvt

warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        svg)

    parameters = edict()

    # path points
    if trainable["point"]:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters


if __name__ == "__main__":

    # cfg = set_config()

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
    h, w = 224, 224

    render = pydiffvg.RenderFunction.apply

    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(
        svg_path="/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/IndieFlower-Regular_a_scaled", trainable={"point": True})

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
        torch.ones(img_init.shape[0], img_init.shape[1],
                   3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]


    # render target image
    shapes_init, shape_groups_init, parameters_init = init_shapes(svg_path="/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/IndieFlower-Regular_e_scaled", trainable={"point": True})
    scene_args_init = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
    img_init_init = render(w, h, 2, 2, 0, None, *scene_args_init)
    img_init_init = img_init_init[:, :, 3:4] * img_init_init[:, :, :3] + \
        torch.ones(img_init_init.shape[0], img_init_init.shape[1],
                    3, device=device) * (1 - img_init_init[:, :, 3:4])
    img_init_init = img_init_init[:, :, :3]
    
    shift_x = alignment(img_init, img_init_init, mode='contact')
    import torchvision
    transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
    img = torch.min(torch.roll(img_init, -(shift_x // 2), 1), torch.roll(transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0), shift_x // 2, 1))

    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, "raster_domain_result.png", gamma=1)

    print("preprocessing")
    # preprocess("IndieFlower-Regular", "Ray", "Ray", 1)
    import svgutils
    # file_1 = '/scratch/gilbreth/zhao969/Word-As-Image/output/dual_if_a/IndieFlower-Regular/a_to_a_dual_if_contact_0.4/output-svg/output_1.svg'
    # file_2 = '/scratch/gilbreth/zhao969/Word-As-Image/output/dual_if_a/IndieFlower-Regular/a_to_a_dual_if_contact_0.4/output-svg/output_2.svg'
    file_1 = "/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/IndieFlower-Regular_a_scaled.svg"
    file_2 = "/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/IndieFlower-Regular_e_scaled.svg"
    svg = svgutils.transform.fromfile(file_2)
    originalSVG = svgutils.compose.SVG(file_2)
    originalSVG.rotate(180, 112, 112)
    originalSVG.move(shift_x // 2, 0)
    figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
    figure.save('New_file2.svg')

    svg = svgutils.transform.fromfile(file_1)
    originalSVG = svgutils.compose.SVG(file_1)
    originalSVG.move(-(shift_x // 2), 0)
    figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
    figure.save('New_file1.svg')

    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        'New_file1.svg')
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg('New_file1.svg', 224, 224, shapes_init, shape_groups_init)

    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        'New_file2.svg')
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        224, 224, shapes_init, shape_groups_init)
    save_svg.save_svg('New_file2.svg', 224, 224, shapes_init, shape_groups_init)

    import svgutils.transform as sg
    import sys
    #create new SVG figure
    fig = sg.SVGFigure(svg.height, svg.width)
    # load matpotlib-generated figures
    fig1 = sg.fromfile('New_file1.svg')
    fig2 = sg.fromfile('New_file2.svg')
    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    # add text labels
    # append plots and labels to figure
    fig.append([plot1, plot2])
    # save generated SVG files
    fig.save("fig_final.svg")
    # combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)