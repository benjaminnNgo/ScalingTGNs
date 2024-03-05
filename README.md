# UTG
Unifying Temporal Graph (UTG) comparison between Continuous Time Dynamic Graphs and Discrete Time Dynamic Graphs 
(merged from UTG_dis, 3/4/2024)


## create branches and development

```
git fetch origin

git checkout -b [branch] origin/[branch]
```


## Workflow

1. DTDG Datasets  (to compare CTDG with DTDG)

```
a. find a discretization level that has no time gap (no empty snapshots in between)
b. generate the negative samples for val and test with `data_script.main_dtdg_gen_ns.py`
c. run experiments with any script starting with `dtdg_` in the root
```

2. TGB Datasets (to compare CTDG with DTDG)

```
a. always continuous and negative samples are downloaded from TGB
b. run experiments with any script starting with `ctdg_` in the root. 
```

3. run CTDG methods with discrete edge timestamps (training edges only)
```
a. generate the discrete timestamps with `data_script.discretize_ctdg_edges.py` will save as `.ts` files
b. load the `.ts` files to remap edge timestamps during training, see `tgn_dtdg_training.py`
c. evaluation is same as TGB
```


## Get Started with DTDG TGX Datasets

use `--wandb` to turn of tracking with wandb


1. TGN commands

```
python dtdg_tgn.py -d enron -t monthly --lr 0.001 --max_epoch 500 --seed 1 --num_run 5 --patience 100

python dtdg_tgn.py -d uci -t weekly --lr 0.001 --max_epoch 500 --seed 1 --num_run 5 --patience 100

python dtdg_tgn.py -d mooc -t daily --lr 0.001 --max_epoch 500 --seed 1 --num_run 5 --patience 100

python dtdg_tgn.py -d social_evo -t daily --lr 0.001 --max_epoch 500 --seed 1 --num_run 5 --patience 100

python dtdg_tgn.py -d contacts -t hourly --lr 0.0001 --max_epoch 200 --seed 1 --num_run 5 --patience 50
```

2. HTGN commands

```
python dtdg_main_htgn.py --model=HTGN --dataset=enron -t monthly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

python dtdg_main_htgn.py --model=HTGN --dataset=uci -t weekly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

python dtdg_main_htgn.py --model=HTGN --dataset mooc -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

python dtdg_main_htgn.py --model=HTGN --dataset social_evo -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

python dtdg_main_htgn.py --model=HTGN --dataset contacts -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50
```

3. EdgeBank Commands

```
python dtdg_edgebank.py -d enron -t monthly --mem_mode unlimited

python dtdg_edgebank.py -d enron -t monthly --mem_mode fixed_time_window
```



## Get Started with TGB Datasets

use `--wandb` to turn of tracking with wandb

```
python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-wiki -t daily --lr 0.0001 --max_epoch 100
```

## Generate negative samples for discrete datasets

```
python main_dtdg_gen_ns.py -d uci -t weekly

python main_dtdg_gen_ns.py -d enron -t monthly

python main_dtdg_gen_ns.py -d mooc -t daily

python main_dtdg_gen_ns.py -d social_evo -t daily
```


## Environment

required dependencies for normal environment

1. install TGB and TGX locally, clone the repo respectively and 
```
pip install -e .
```

2. install torch, PyG and other dependencies

```
pip install torch==1.13.1
pip install torch-geometric==2.3.1 
pip install torch-scatter==2.1.1
pip install geoopt
pip install -r requirements.txt
```

for compute canada, use:
```
pip install -r requirements_ccai.txt
```


### Running discretized edges for CTDG methods

1. convert CTDG edgelist into DTDG edgelist (for the training set)

2. store the converted DTDG edgelist (with converted UNIX timestamps)

3. load the DTDG training set with TGB framework (or construct separate data loading / data class)

4. train TGN on DTDG training set

5. use the TGB class for evaluation set (the test edges) and evaluation