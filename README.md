<p align="center">
  <img width="300" height="150" src="https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/TGS_Logo.png">
</p>

# Towards Neural Scaling Laws for Foundation Models on Temporal Graphs

This repository provides the implementation of the TGS foundation model benchmarking and includes links to temporal networks suitable for foundation model training. TGS introduces a training process for foundation models using various real-world temporal networks, enabling prediction on previously unseen networks.

## Overview
Temporal graph learning focuses on predicting future interactions from evolving network data. Our study addresses whether it's possible to predict the evolution of an unseen network within the same domain using observed temporal graphs. We introduce the Temporal Graph Scaling (TGS) dataset, comprising 84 ERC20 token transaction networks collected from 2017 to 2023. To evaluate transferability, we pre-train Temporal Graph Neural Networks (TGNNs) on up to 64 token transaction networks and assess their performance on 20 unseen token types. Our findings reveal that the neural scaling law observed in NLP and Computer Vision also applies to temporal graph learning: pre-training on more networks with more parameters enhances downstream performance. This is the first empirical demonstration of temporal graph transferability. Notably, the largest pre-trained model surpasses fine-tuned TGNNs on unseen test networks, marking a significant step towards building foundation models for temporal graphs. The code and datasets are publicly available.

![](https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/Figure4.jpg)
*TGS foundation model performance on unseen networks*

### Dataset and Benchmark Implementation 
All extracted transaction networks required for multi-network model training can be downloaded [here](#).

Link has been removed for the purpose of anonymizing the authors. 


The TGS dataset and benchmark include: 
(1) Token extraction: extracting the token transaction network from our P2P Ethereum live node. 
(2) Discretization: creating weekly snapshots to form discrete time dynamic graphs. 
(3) Foundation Model Training: TGS transaction networks are divided randomly into train and test sets. We train the MNs on a collection of training networks. Lastly, MNs are tested on 20 unseen test networks.

![](https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/img_2.jpg)
*TGS Dataset and Benchmark Overview*

### About datasets
- Each ```.csv``` file represents all transactions of the token network that has the same name as the file name (```tokenname.csv```)
- Each transaction corresponds to a row in each file
- The information of each transaction is recorded as the table below:

| column name | meaning|
|-------------|----------------------------------------------------------------------------------------------------------------------------|
| blockNumber | is the block ID of Ethereum that includes this transaction 2                                                              |
| timestamp   | time that the transaction is made in UNIX timestamp format                                                                |
| tokenAddress | the address that specifies a unique ERC20 token                                                                            |
| from        | address of sender                                                                                                         |
| to          | address of receiver                                                                                                        |
| value       | the amount the transaction                                                                                                 |
| fileBlock   | we split the whole number of blocks count to 35 buckets and assigned the bucket ID to the transaction to trace the blocks  |

- To use the same setting as described in the papers, we include edge list and label that contain node interactions and labels for each snapshot in each token network
  -  Each transaction in the edge list also has "from","to" and "amount" fields, but with an additional "snapshot" field to indicate the index of the snapshot that the transaction below to
  -  Each row in label file indicates the ground truth label of the snapshot having an index corresponding to the index of the row (e.g first row indicates the label of the first snapshot)
- However, we also provide raw ```.csv```  to divide into generate edges list and label with a different setting.


# Core backbone package installation

1. install torch

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. install PyG

```
pip install torch_geometric==2.4.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

3. install PyTorch Geonetric Temporal (optional)

```
pip install torch-geometric-temporal
```
### Prerequisites

- Python 3.8+
- Libraries listed in `installed_packages.txt`

# Results reproduce
To train a single or multi-network model, download datasets from [here](#).

Link has been removed for the purpose of anonymizing the authors. 

- All label files need to be placed in the directory `data/input/raw/labels/ `
- All edge list files need to be placed in the directory `data/input/raw/edgelists/ `
- All raw `.csv` files need to be placed in the directory `data/input/tokens/raw/ ` if you want to re-generate edge lists and labels.

## Multi-network Models
### Data Loader
Multi-network loading for  foundation model training is done through the following function which is already included in the `train_foundation_tgc.py` and `test_foundation_tgc.py` scripts.
```
load_multiple_datasets("dataset_package_2.txt")
```

### Model Training
To train the multi-network model `train_foundation_tgc.py` should be used. Examples include:
```
python train_foundation_tgc.py --model=HTGN --max_epoch=300 --lr=0.0001 --seed=710 --wandb
```
### Model Inference
In order to inference testing on saved multi-network models `test_foundation_tgc.py` is used:

```
python test_foundation_tgc.py --model=HTGGN --seed=710
```

## Single Model
- To train a single model, run  `train_single_tgc.py` inside `/script/`. Hyper-parameters can easily be configured by modifying `args.{parameter_name}` inside the file.
- It is also possible to run the code and set hyper-parameter by using the commands. Example:
```
python train_single_tgc.py --model=HTGN --max_epoch=300 --lr=0.0001 --seed=710 --wandb
```
*Make sure to comment out following chunk of code to avoid over-writing when you use the commands to run the code*
```
args.max_epoch = 250
args.wandb = False #Set this to true if you want to use wandb as a training debug tool
args.min_epoch = 100
args.model = "HTGN"
args.log_interval = 10
args.lr = 0.00015
args.patience = 20
```


