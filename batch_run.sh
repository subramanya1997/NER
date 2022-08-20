#!/bin/sh
#SBATCH --job-name=dylog
#SBATCH -o /work/snagabhushan_umass_edu/dylog/logs/train_%j.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=128GB  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH --gres gpu 
#SBATCH -G 1  # Number of GPUs
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

module load cuda/11.7.0


conda activate py39

cd /work/snagabhushan_umass_edu/dylog
# python infer.py -p fix_infer -w orig --load-path /work/shantanuagar_umass_edu/ego4d/model/nlq/meme_long_6_last.pth >sbatch/out/fix_infer.txt
# python train.py -w orig >sbatch/out/fix_infer.txt

python train.py > logs/train_orig_%j.txt