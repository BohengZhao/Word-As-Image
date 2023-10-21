#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

set -e

USE_WANDB=1 # CHANGE IF YOU WANT WANDB
WANDB_USER="zhao969"

# tone_plus_conformal, tone, conformal, none, trocr, font_loss, font_loss_upgrade, dual_font_loss
EXPERIMENT=dual_if_whole_word # _dream # _font_style

WORDS=(vision learning)
#TARGET=e
TARGETS=("e")
#TARGETS=("y" "h" "l")
letter_=("E")
fonts=(IndieFlower-Regular)
#styles=(Arial Curly Artistic Plain)
styles=(plain)
alignment=(overlap)
#fonts=(IndieFlower-Regular Quicksand LuckiestGuy-Regular)
for j in "${fonts[@]}"
do
    SEED=0
    for i in "${letter_[@]}"
    do
        for target in "${TARGETS[@]}"
        do 
            for weight in 0.5
            do
                for WORD in "${WORDS[@]}"
                do 
                    echo "$i"
                    font_name=$j
                    ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${font_name} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --init_char ${i} --target_char ${target}"
                    CUDA_VISIBLE_DEVICES=0 python code/whole_word_opt.py $ARGS --word "${WORD}" --dual_bias_weight_sweep "${weight}"
                done
            done
        done
    done
done