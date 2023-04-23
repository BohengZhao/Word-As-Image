submit: 
	sbatch --nodes=1 --gpus-per-node=1 --mem=30G -A standby jobscript.sh

watch:
	LOGFILES=$(shell ls slurm-* | tail -1); tail -f $$LOGFILES

clean:
	rm -rf output/
	rm slurm-*
	mkdir output

