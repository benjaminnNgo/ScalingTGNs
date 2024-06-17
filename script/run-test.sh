#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH --output=../run_scripts/dt%j.txt
#SBATCH --error=../run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=24                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 titan xp
#SBATCH --mem=60G                             # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                       # The job will run for 1 day

export HOME="/home/mila/r/razieh.shirzadkhani/ScalingTGNs"
module load python/3.10
source /home/mila/r/razieh.shirzadkhani/tgnn/bin/activate



pwd
python test_HTGN_foundation.py 
# python gclstm_test_foundation_model.py