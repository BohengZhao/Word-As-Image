#!/bin/bash

# source /home/zhao969/.bashrc
# conda activate ambigen
# cd $SLURM_SUBMIT_DIR

# python code/test.py

#!/bin/bash
source /home/zhao969/.bashrc
conda activate ambigen
cd $SLURM_SUBMIT_DIR

# zip -r -q output_folder_second_font.zip ./output
# python code/html_select.py
python code/main_align.py 
# python code/whole_word_opt.py