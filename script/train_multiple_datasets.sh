#!/bin/bash

directory="../data/input/raw/edgelists"
model="HTGN"
space="==============================="
seed=710
echo "$directory"

# Loop through all files in the directory
for file in "$directory"/*; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
        # Get the base name of the file
        base_name="${file##*/}"
        # Get the suffix part of the base name
        dataset="${base_name%%_*}"
        echo "$space Train on $dataset $space"
        # Print the suffix
        echo "python train_tgc_end_to_end.py --model=$model --seed=$seed --dataset=$dataset"
        python train_tgc_end_to_end.py --model="$model" --seed=$seed --dataset="$dataset"
    fi
done