import argparse
import os.path as osp
import yaml
import random
from easydict import EasyDict as edict
import numpy.random as npr
import torch
from utils import (
    edict_2_dict,
    check_and_create_dir,
    update)
import wandb
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config/base.yaml")
    parser.add_argument("--experiment", type=str,
                        default="conformal_0.5_dist_pixel_100_kernel201")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--log_dir', metavar='DIR', default="output")
    parser.add_argument('--font', type=str, default="none", help="font name")
    parser.add_argument('--word', type=str, default="none",
                        help="the text to work on")
    parser.add_argument('--optimized_letter', type=str,
                        default="none", help="the letter in the word to optimize")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_user', type=str, default="none")
    parser.add_argument('--init_char', type=str, default="none")
    parser.add_argument('--target_char', type=str, default="none")
    parser.add_argument('--alignment', type=str, default="none")
    parser.add_argument('--dual_bias_weight_sweep', type=float, default=1.0)
    parser.add_argument('--post_processing', type=str, default="dual-svg")
    parser.add_argument('--word_group', type=int, default=0)
    #parser.add_argument('--angle_w_sweep', type=float, default=0.3)
    #parser.add_argument('--font_loss_weight', type=float, default=100.0)
    #parser.add_argument('--sweep_lr_base', type=float, default=1.0)
    #parser.add_argument('--sweep_lr_init', type=float, default=0.02)
    #parser.add_argument('--sweep_lr_final', type=float, default=0.0008)


    cfg = edict()
    args = parser.parse_args()
    with open('TOKEN', 'r') as f:
        setattr(args, 'token', f.read().replace('\n', ''))
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.font = args.font
    cfg.word = args.word
    cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
    cfg.optimized_letter = args.optimized_letter
    cfg.batch_size = args.batch_size
    cfg.token = args.token
    cfg.use_wandb = args.use_wandb
    cfg.wandb_user = args.wandb_user
    cfg.letter = f"{args.font}_{args.optimized_letter}_scaled"
    cfg.target = f"code/data/init/{cfg.letter}"
    cfg.init_char = args.init_char
    cfg.target_char = args.target_char
    cfg.init_letter = f"{args.font}_{args.target_char}_scaled"
    cfg.init = f"code/data/init/{cfg.init_letter}"
    cfg.dual_bias_weight_sweep = args.dual_bias_weight_sweep
    cfg.alignment = args.alignment
    cfg.post_processing = args.post_processing
    cfg.word_group = args.word_group
    
    #cfg.angle_w_sweep = args.angle_w_sweep
    #cfg.font_loss_weight_sweep = args.font_loss_weight
    #cfg.sweep_lr_base = args.sweep_lr_base
    #cfg.sweep_lr_init = args.sweep_lr_init
    #cfg.sweep_lr_final = args.sweep_lr_init / 100.0

    return cfg


def set_config():

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    # recursively traverse parent_config pointers in the config dicts
    cfg_key = cfg_arg.experiment
    cfgs = [cfg_arg]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get('parent_config', 'baseline')

    # allowing children configs to override their parents
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    # set experiment dir
    if cfg.word == "none":
        signature = f"{cfg.init_char}_to_{cfg.target_char}_{cfg.experiment}_{cfg.alignment}" #_{cfg.dual_bias_weight_sweep}
        cfg.experiment_dir = \
            osp.join(cfg.log_dir, cfg.font, signature)
        configfile = osp.join(cfg.experiment_dir, 'config.yaml')
        print('Config:', cfg)
    else:
        signature = f"whole_word_opt_{cfg.word_group}" #_{cfg.dual_bias_weight_sweep}
        cfg.experiment_dir = \
            osp.join(cfg.log_dir, cfg.font, signature)
        configfile = osp.join(cfg.experiment_dir, 'config.yaml')
        print('Config:', cfg)

    # create experiment dir and save config
    # check_and_create_dir(configfile)
    # with open(osp.join(configfile), 'w') as f:
    #     yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(project="Word-As-Image", entity=cfg.wandb_user,
                   config=cfg, name=f"{signature}", id=wandb.util.generate_id())

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        assert False

    return cfg
