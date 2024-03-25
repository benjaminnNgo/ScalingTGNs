#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=../run_scripts/dt%j.txt
#SBATCH --error=../run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=2                    # Ask for 4 CPUs
#SBATCH --gres=gpu:1                         # Ask for 1 titan xp
#SBATCH --mem=32G                             # Ask for 32 GB of RAM
#SBATCH --time=100:20:00                       # The job will run for 1 day

export HOME="/home/mila/r/razieh.shirzadkhani/ScalingTGNs"
module load python/3.10
source /home/mila/r/razieh.shirzadkhani/tgnn/bin/activate



pwd
python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_foundation.py --model=HTGN --seed=710 --dataset=first_test
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/main.py --model=HTGN --seed=710 --dataset=aion