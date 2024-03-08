#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=htgn_contacts_hourly.txt 
#SBATCH --error=htgn_contacts_hourly_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=72:00:00    #48:00:00                   # The job will run for 1 day

export HOME="/home/mila/h/huangshe"
module load python/3.9
source $HOME/tgbenv/bin/activate
pwd


# CUDA_VISIBLE_DEVICES=0 python dtdg_main_htgn.py --model=HTGN --dataset=enron -t monthly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_main_htgn.py --model=HTGN --dataset=uci -t weekly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_main_htgn.py --model=HTGN --dataset mooc -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_main_htgn.py --model=HTGN --dataset social_evo -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

CUDA_VISIBLE_DEVICES=0 python dtdg_main_htgn.py --model=HTGN --dataset contacts -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_training.py -d tgbl-coin -t weekly --seed 3
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-wiki -t hourly --seed 3 --dtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-wiki -t hourly --seed 1 --nodtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-review -t minutely --seed 1 --nodtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-review -t hourly --seed 1 --dtrain