#!/bin/bash

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=if_loss_upgrade

CONCEPT="none"
WORD=h
TARGET=y
fonts=(IndieFlower-Regular)
for j in "${fonts[@]}"
do
    letter_=("h")
    SEED=0
    for i in "${letter_[@]}"
    do
        #for angle_w_sweep in 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.01
        #for font_weight_sweep in 0.1 0.05 0.01 0.005 0.001
        #do 
            echo "$i"
            font_name=$j
            ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --init_char ${i} --target_char ${TARGET}"
            CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --word "${WORD}"
        #done
    done
done