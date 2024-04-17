#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=../run_scripts/dt%j.txt
#SBATCH --error=../run_scripts/dt%jerror.txt 
#SBATCH --cpus-per-task=5                  # Ask for 4 CPUs
#SBATCH --gres=gpu:1                       # Ask for 1 titan xp
#SBATCH --mem=20G                             # Ask for 32 GB of RAM
#SBATCH --time=10:20:00                       # The job will run for 1 day

export HOME="/home/mila/r/razieh.shirzadkhani/ScalingTGNs"
module load python/3.10
source /home/mila/r/razieh.shirzadkhani/tgnn/bin/activate



pwd
# python BaselineProcess.py
python train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21655_0xbcca60bb61934080951369a648fb03df4f96263c.csv --testlength=30 --log_interval=5 --wandb --max_epoch=100
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_foundation.py --model=HTGN --seed=710 --dataset=.. --wandb --max_epoch=100
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/main.py --model=HTGN --seed=710 --dataset=aion
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21658_0x5f98805a4e8be255a32880fdec7f6728c6568ba0.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21654_0x09a3ecafa817268f77be1283176b946c4ff2e608.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21655_0xbcca60bb61934080951369a648fb03df4f96263c.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21639_0x1ceb5cb57c4d4e2b2433641b95dd330a33185a44.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21624_0x83e6f1e41cdd28eaceb20cb649155049fac3d5aa.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_1898_0x00a8b738e453ffd858a7edf03bccfe20412f0eb0.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21630_0xcc4304a31d09258b0029ea7fe63d032f52e44efe.csv
# python /home/mila/r/razieh.shirzadkhani/ScalingTGNs/script/train_tgc_end_to_end.py --model=HTGN --seed=710 --dataset=unnamed_token_21636_0xfca59cd816ab1ead66534d82bc21e7515ce441cf.csv