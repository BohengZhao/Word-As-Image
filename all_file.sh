for file in /scratch/gilbreth/zhao969/Word-As-Image/scripts/*
do
    sbatch --nodes=1 --gpus-per-node=1 --mem=80G -A standby --time=4:00:00 --constraint "a100-80gb" --exclude=gilbreth-k019,gilbreth-k025 "$file" 
done