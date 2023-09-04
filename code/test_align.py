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

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    word = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
    # preprocess(cfg.font, word, cfg.optimized_letter, cfg.level_of_cc)

    h, w = 224, 224

    data_augs = get_data_augs(224)

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
    mask1 = img_init[:, :, :3]
    img_init = img_init[:, :, :3]


    # render target image
    shapes_init, shape_groups_init, parameters_init = init_shapes(svg_path=cfg.init, trainable=cfg.trainable)
    scene_args_init = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
    img_init_init = render(w, h, 2, 2, 0, None, *scene_args_init)
    img_init_init = img_init_init[:, :, 3:4] * img_init_init[:, :, :3] + \
        torch.ones(img_init_init.shape[0], img_init_init.shape[1],
                    3, device=device) * (1 - img_init_init[:, :, 3:4])
    mask2 = img_init_init[:, :, :3]
    img_init_init = img_init_init[:, :, :3]

    shift_x_sep = alignment(mask1, mask2, mode='seperated')
    transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
    img = torch.min(torch.roll(img_init, -(shift_x_sep // 2), 1), torch.roll(transform(img_init_init.permute(2, 0, 1)).permute(1, 2, 0), shift_x_sep // 2, 1))
    plt.imsave("align.png", img.detach().cpu().numpy())
    print("Finish aligning.")
