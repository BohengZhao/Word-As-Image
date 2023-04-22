#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

bash run_word_as_image.sh
