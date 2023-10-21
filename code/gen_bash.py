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
    render_svgs_all_type,
    gen_bash_script,
    gen_bash_all,
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

def gen_whole_word():
    words = []
    with open("/scratch/gilbreth/zhao969/BenchMark/words.txt", "r") as input_file:
        for line in input_file:
            word = line.strip()
            words.append(word)
    
    template = '''#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=dual_if_whole_word # _dream # _font_style

WORDS=(whole)
#TARGET=e
TARGETS=("e")
#TARGETS=("y" "h" "l")
letter_=("E")
fonts=(IndieFlower-Regular)
#styles=(Arial Curly Artistic Plain)
styles=(plain)
alignment=(overlap)
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
for j in "${{fonts[@]}}"
do
    SEED=0
    for i in "${{letter_[@]}}"
    do
        for target in "${{TARGETS[@]}}"
        do 
            for weight in 0.5
            do
                for WORD in "${{WORDS[@]}}"
                do 
                    echo "$i"
                    font_name=$j
                    ARGS="--experiment $EXPERIMENT --optimized_letter ${{i}} --seed $SEED --font ${{font_name}} --use_wandb ${{USE_WANDB}} --wandb_user ${{WANDB_USER}} --init_char ${{i}} --target_char ${{target}}"
                    CUDA_VISIBLE_DEVICES=0 python code/whole_word_opt.py $ARGS --word "${{WORD}}" --dual_bias_weight_sweep "${{weight}}" --word_group {idx}
                done
            done
        done
    done
done'''
    group_size = len(words) // 20
    groups = [words[i:i + group_size] for i in range(0, len(words), group_size)]
    for idx in range(len(groups)):
        with open (f'/scratch/gilbreth/zhao969/Word-As-Image/whole_word_scripts/run_{idx}.sh', 'w') as bashfile:
            words_string = ' '.join(groups[idx])
            bashfile.write(template.format(idx=idx))

if __name__ == "__main__":

    # use GPU if available
    # gen_bash_script("warm", "warm", weight_sweep=[0.4, 0.45, 0.5, 0.55, 0.6], alignment=["none", "overlap", "seperated_2", "seperated"], font="IndieFlower-Regular")
    # gen_bash_all(weight_sweep=[0.4, 0.45, 0.5, 0.55, 0.6], alignment=["none", "overlap", "seperated_2", "seperated"], font="IndieFlower-Regular")
    # gen_bash_all(weight_sweep=[0.4, 0.5, 0.6], alignment=["none", "overlap", "seperated_2", "seperated"], font="Quicksand")
    gen_whole_word()