#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=tgn_coin_minute_s1.txt 
#SBATCH --error=tgn_coin_minute_s1_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=64G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00                       # The job will run for 1 day

export HOME="/home/mila/h/huangshe"
module load python/3.9
source $HOME/tgbenv/bin/activate
pwd


CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_training.py -d tgbl-coin -t minutely --seed 3 

