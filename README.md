<p align="center">
  <img width="300" height="150" src="https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/TGS_Logo.png">
</p>

# Towards Neural Scaling Laws for Foundation Models on Temporal Graphs

This repository provides the implementation of the TGS foundation model benchmarking and includes links to temporal networks suitable for foundation model training. TGS introduces a training process for foundation models using various real-world temporal networks, enabling prediction on previously unseen networks.

## Overview
Temporal graph learning focuses on predicting future interactions from evolving network data. Our study addresses whether it's possible to predict the evolution of an unseen network within the same domain using observed temporal graphs. We introduce the Temporal Graph Scaling (TGS) dataset, comprising 84 ERC20 token transaction networks collected from 2017 to 2023. To evaluate transferability, we pre-train Temporal Graph Neural Networks (TGNNs) on up to 64 token transaction networks and assess their performance on 20 unseen token types. Our findings reveal that the neural scaling law observed in NLP and Computer Vision also applies to temporal graph learning: pre-training on more networks with more parameters enhances downstream performance. This is the first empirical demonstration of temporal graph transferability. Notably, the largest pre-trained model surpasses fine-tuned TGNNs on unseen test networks, marking a significant step towards building foundation models for temporal graphs. The code and datasets are publicly available.

![](https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/htgn-log2-all-v3-1.png)
*TGS foundation model performance on unseen networks*

### Dataset
All extracted transaction networks required for foundation model training can be downloaded [here](https://zenodo.org/doi/10.5281/zenodo.11455827).

The standard ML croissant repository for datasets TGS's metadata is also available [here](https://huggingface.co/datasets/ntgbaoo/Temporal_Graph_Scaling_TGS_Benchmark).

The TGS dataset extraction includes: 
(1) Token Extraction: extracting the token transaction network from our P2P Ethereum live node. 
(2) Discretizing: creating weekly snapshots for the Discretized Temporal Directed Graph (DTDG) setting. 
(3) Labeling: assigning labels based on network growth; increasing trends are labeled one, decreasing trends are labeled zero.

![](https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/Data_Processing_V1.png)
*TGS dataset extraction*

### Benchmark Implementation

 TGS transaction networks are divided randomly into train and test sets. The train set is used to train foundation models with different sizes; then, the trained models are evaluated on the test set.

![](https://github.com/benjaminnNgo/ScalingTGNs/blob/main/pic/Foundation_training_vf.png)
*TGS foundation model training overview*

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
1. Single models' results can be reproduced on all datasets provided by TGS by running ```train_tgc_end_to_end.py```
2. To reproduce the foundation model train and test results, please follow step:








