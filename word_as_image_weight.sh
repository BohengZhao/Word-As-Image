#!/bin/bash

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=dual_if 

CONCEPT="none"
WORD=h # No effect for this experiment
TARGETS=("y" "k" "o")
letter_=("h")
fonts=(IndieFlower-Regular)
#styles=(Arial Curly Artistic Plain)
styles=(plain)
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
                #for weight in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
                for weight in 0.51 #0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59
                do 
                    echo "$i"
                    font_name=$j
                    ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --init_char ${i} --target_char ${target}"
                    CUDA_VISIBLE_DEVICES=0 python code/main_dual_sweep.py $ARGS --semantic_concept "${CONCEPT}" --word "${WORD}" --target_style "${style}" --init_style "${style}" --dual_bias_weight_sweep "${weight}"
                done
            done
        done
    done
done