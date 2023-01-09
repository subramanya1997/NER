#!/bin/sh
#SBATCH --job-name=dylog
#SBATCH -o /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/NER/logs/logs.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=128GB  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH --gres gpu 
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

module load cuda/11.6.0

conda activate py39

cd /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/NER

python train.py > logs/train_orig.txt