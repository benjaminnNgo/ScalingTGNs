#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=../run_scripts/dt%j.txt
#SBATCH --error=../run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=4                 # Ask for 4 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 titan xp
#SBATCH --mem=20G                             # Ask for 32 GB of RAM
#SBATCH --time=168:00:00                     

export HOME="/home/mila/r/razieh.shirzadkhani/ScalingTGNs"
module load python/3.10
source /home/mila/r/razieh.shirzadkhani/tgnn/bin/activate



pwd
# python gclstm_foundation_model.py --wandb
python train_foundation_tgc_v2.py --wandb
# python train_tgc_end_to_end.py
