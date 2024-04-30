#!/bin/bash
data_pack_file="../data/dataset_pack_original"
model="HTGN"
space="==============================="

seeds=(710 720 800)

if [ -f "$data_pack_file" ]; then
    # Declare an empty array to store the lines
    datasets=()

    # Read the file line by line
    while IFS= read -r line; do
        # Append the line to the array
        line="${line//$'\n'/}"
        datasets+=("$line")
    done < "$data_pack_file"
    for dataset in "${datasets[@]}"
    do
      for seed in "${seeds[@]}"
      do
        echo "$space"
        echo python train_tgc_end_to_end.py --model=$model --seed=$seed --log_interval=20 --wandb --max_epoch=250 --lr=0.00015 --patience=20 --dataset=$dataset
      done
    done

else
    echo "File not found: $file_path"
fi

# Loop over the array and print each element
