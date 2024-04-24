import math
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt
import wandb
import warnings

def extra_dataset_attributes_loading(dataset):

    partial_path = f'../data/input/raw/'

    # load graph labels
    label_filename = f'{partial_path}/labels/{dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)


    return TG_labels

def extra_dataset_baseLine_label(dataset):
    partial_path = f'../data/input/raw/'
    # load baseline model labels
    baseline_label_filename = f'{partial_path}/labels/{dataset}_dummy_fd_ld_labels.csv'
    baseline_label_df = pd.read_csv(baseline_label_filename, header=None, names=['label'])
    baseline_labels = torch.from_numpy(np.array(baseline_label_df['label'].tolist())).to(args.device)
    return baseline_labels

class BaseLineModel:
    def __init__(self,data,dataset_name):
        test_ratio = 0.15
        self.len = data['time_length']
        testlength = math.floor(self.len * test_ratio)  # Re-calculate number of test snapshots
        self.start_train = 0
        self.train_shots = list(range(self.start_train, self.len - testlength*2)) #exclude validation sets and test sets
        self.test_shots = list(range(self.len - testlength, self.len))
        self.t_graph_labels = extra_dataset_attributes_loading(dataset_name)
        self.baseline_labels = extra_dataset_baseLine_label(dataset_name)

    def test(self):
        tg_labels = []
        for t_eval_idx, t in enumerate(self.test_shots):
            tg_labels.append(self.t_graph_labels[t_eval_idx + len(self.train_shots)].cpu().numpy())

        tg_preds = []
        for t_eval_idx, t in enumerate(self.test_shots):
            tg_preds.append(self.baseline_labels[t_eval_idx + len(self.train_shots)].cpu().numpy())



        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap



if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare
    datasets_list_name = [
        'unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0',
        'unnamedtoken216240x83e6f1e41cdd28eaceb20cb649155049fac3d5aa',
        'unnamedtoken216300xcc4304a31d09258b0029ea7fe63d032f52e44efe',
        'unnamedtoken216350xe53ec727dbdeb9e2d5456c3be40cff031ab40a55',
        'unnamedtoken216360xfca59cd816ab1ead66534d82bc21e7515ce441cf',
        'unnamedtoken216390x1ceb5cb57c4d4e2b2433641b95dd330a33185a44',
        'unnamedtoken216540x09a3ecafa817268f77be1283176b946c4ff2e608',
        'unnamedtoken216550xbcca60bb61934080951369a648fb03df4f96263c',
        'unnamedtoken216580x5f98805a4e8be255a32880fdec7f6728c6568ba0',
        'unnamedtoken216620x429881672b9ae42b8eba0e26cd9c73711b891ca5'
    ]
    columns = ['dataset','auc','ap']
    rowlist = []
    for dataname in datasets_list_name:
        row = []
        data = loader(dataset=dataname, neg_sample="rnd")
        baseModel = BaseLineModel(data,dataname)
        auc, ap = baseModel.test()
        row.append(dataname)
        row.append(auc)
        row.append(ap)
        rowlist.append(row)

    df = pd.DataFrame(rowlist,columns= columns)
    df.to_csv('../data/output/baselinemodel.csv', index=False)
