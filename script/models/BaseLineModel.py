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

from script.utils.BaselineProcess import createDummyBaselineLabelsWeekly


def extra_dataset_attributes_loading(dataset):

    partial_path = f'../../data/input/raw/'

    # load graph labels
    label_filename = f'{partial_path}/labels/{dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)


    return TG_labels

def extra_dataset_baseLine_label(dataset):
    partial_path = f'../../data/input/raw'
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
    from script.utils.data_util import loader
    csv_lists = [
        # 'unnamed_token_21403_0x07e0edf8ce600fb51d44f51e3348d77d67f298ae.csv',
        # 'unnamed_token_18716_0x06da0fd433c1a5d7a4faa01111c044910a184553.csv',
        # 'CRO_0xa0b73e1ff0b80914ab6fe0444e65848c4c34450b.csv',
        'Gemini_dollar_0x056fd409e1d7a124bd7017459dfea2f387b6d5cd.csv',
        'CVC_0x41e5560054824ea6b0732e656e3ad64e20e94e45.csv'
    ]

    result_path = '../../data/output/baselinemodel.csv'
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=['dataset','auc','ap'])
    else:
        result_df = pd.read_csv(result_path)

    for csv_file_name in csv_lists:
        try:
            dataname = csv_file_name.replace("_","").replace(".csv","")
            data = loader(dataset=dataname, neg_sample="rnd")
            createDummyBaselineLabelsWeekly(csv_file_name)
            baseModel = BaseLineModel(data,dataname)
            auc, ap = baseModel.test()


            result_df = result_df._append({'dataset': dataname,
                                           'auc': auc,
                                           'ap': ap,
                                           }, ignore_index=True)
            result_df.to_csv(result_path, index=False)
        except Exception as e:
            print("Can't process csv file {}".format(csv_file_name))
