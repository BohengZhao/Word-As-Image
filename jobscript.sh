#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

#bash run_word_as_image.sh
bash sweep_word_as_image.sh
#bash sweep_all_pairs_6.sh
#bash word_as_image_weight.sh
#cd code
#python IF_pipe.py