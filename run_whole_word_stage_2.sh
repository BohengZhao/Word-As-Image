#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=dual_if_whole_word_stage_2

WORD=the
TARGET=y
fonts=(IndieFlower-Regular)
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
for j in "${fonts[@]}"
do
    letter_=("h")
    SEED=0
    for i in "${letter_[@]}"
    do
        echo "$i"
        font_name=$j
        ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --init_char ${i} --target_char ${TARGET}"
        CUDA_VISIBLE_DEVICES=0 python code/whole_word_stage_2.py $ARGS --word "${WORD}" --dual_bias_weight_sweep 0.75
    done
done