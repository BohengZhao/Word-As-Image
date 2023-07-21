submit: 
	sbatch --nodes=1 --gpus-per-node=1 --mem=30G -A standby --time=4:00:00 --constraint "a100-80gb|a100-40gb" jobscript.sh

watch:
	LOGFILES=$(shell ls slurm-* | tail -1); tail -f $$LOGFILES

clean:
	rm -rf output/
	rm slurm-*
	mkdir output

