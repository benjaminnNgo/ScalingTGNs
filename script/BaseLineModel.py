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

    # load graph lables
    label_filename = f'{partial_path}/labels/{dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = label_df['label'].tolist()
    return TG_labels

class BaseLineModel:
    def __init__(self,data,dataset_name):
        test_ratio = 0.15
        self.len = data['time_length']
        testlength = math.floor(self.len * test_ratio)  # Re-calculate number of test snapshots
        self.start_train = 0
        self.train_shots = list(range(self.start_train, self.len - testlength))
        self.test_shots = list(range(self.len - testlength, self.len))
        self.t_graph_labels = extra_dataset_attributes_loading(dataset_name)

    def test(self):
        tg_labels = [self.t_graph_labels[i] for i in self.test_shots]
        tg_preds = [0]
        for i in range(1,len(self.t_graph_labels)-1):
            tg_preds.append(self.t_graph_labels[i])

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap



if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare
    data = loader(dataset="unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0", neg_sample="rnd")
    baseModel = BaseLineModel(data,"unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0")
    auc, ap = baseModel.test()
    print(auc, ap)