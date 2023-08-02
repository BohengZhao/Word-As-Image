#!/bin/bash

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=dual_if

CONCEPT="none"
WORD=h # No effect for this experiment
#TARGET=e
TARGETS=("y" "Y" "k" "K" "e" "E" "l" "L" "h" "H")
letter_=("h")
fonts=(IndieFlower-Regular)
#styles=(Arial Curly Artistic Plain)
styles=(arial)
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
for j in "${fonts[@]}"
do
    SEED=0
    for i in "${letter_[@]}"
    do
        for style in "${styles[@]}"
        do 
            for target in "${TARGETS[@]}"
            do 
                echo "$i"
                font_name=$j
                ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --init_char ${i} --target_char ${target}"
                CUDA_VISIBLE_DEVICES=0 python code/main_dual.py $ARGS --semantic_concept "${CONCEPT}" --word "${WORD}" --target_style "${style}" --init_style "${style}"
            done
        done
    done
done