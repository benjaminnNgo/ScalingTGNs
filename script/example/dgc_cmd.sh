#!/bin/bash

# model_name="GRUGCN"
# model_name="EGCN"
model_name="HTGN"
max_epoch=200

seed=1
dataset_name="aion"



if [ "$model_name"="EGCN" ]
then
    echo "=========================================="
    echo " >>> seed:    $seed"
    echo " >>> MODEL:   $model_name"
    echo " >>> DATA:    $dataset_name"
    echo "=========================================="

    python baselines/run_evolvegcn_baselines_TGC.py --models "$model_name" --seed "$seed"  \
    --dataset "$dataset_name" --max_epoch "$max_epoch" 
    echo "=========================================="
    echo "=========================================="

elif [ "$model_name"="GRUGCN" ]
then
    echo "=========================================="
    echo " >>> Seed: $seed"
    echo " >>> MODEL: $model_name"
    echo " >>> DATA: $dataset_name"
    echo "=========================================="
    python train_tgc_end_to_end.py --models "$model_name" --seed "$seed"  \
    --dataset "$dataset_name" --max_epoch "$max_epoch" 
    echo "=========================================="
    echo "=========================================="

elif [ "$model_name"="HTGN" ]
then
    echo "=========================================="
    echo " >>> Seed: $seed"
    echo " >>> MODEL: $model_name"
    echo " >>> DATA: $dataset_name"
    echo "=========================================="
    python train_tgc_end_to_end.py --models "$model_name" --seed "$seed"  \
    --dataset "$dataset_name" --max_epoch "$max_epoch" 
    echo "=========================================="
    echo "=========================================="

else 
    echo "Undefined Model Name!!!"
fi
