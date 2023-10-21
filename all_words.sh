for i in $(seq 0 19); do
	sbatch --nodes=1 --gpus-per-node=1 --mem=80G -A standby --time=4:00:00 --exclude=gilbreth-k019,gilbreth-k025,gilbreth-k009 --constraint "a100-80gb" whole_word_scripts/run_${i}.sh
done