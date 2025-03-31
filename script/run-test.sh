#!/bin/bash
#SBATCH --job-name=gclstm-2
#SBATCH --partition=main
#SBATCH --output=../run_scripts/dt%j.txt
#SBATCH --error=../run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=4                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 titan xp
#SBATCH --mem=20G                             # Ask for 32 GB of RAM
#SBATCH --time=20:00:00                       # The job will run for 1 day

export HOME="/home/mila/r/razieh.shirzadkhani/ScalingTGNs"
module load python/3.10
source /home/mila/r/razieh.shirzadkhani/tgnn/bin/activate



pwd
python test_foundation_tgc.py 
# python gclstm_test_foundation_model.py