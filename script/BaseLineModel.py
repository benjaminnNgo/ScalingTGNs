import math
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from script.utils.TGS import TGS_Handler


def extra_dataset_attributes_loading(dataset):

    partial_path = f'../data/input/raw/'

    # load graph labels
    label_filename = f'{partial_path}/labels/{dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)


    return TG_labels

def extra_dataset_baseLine_label(dataset):
    partial_path = f'data/input/raw/dummy_labels'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    # load baseline model labels
    baseline_label_filename = f'{partial_path}/{dataset}_dummy_fd_ld_labels.csv'
    print(baseline_label_filename)
    print(os.path.exists(baseline_label_filename))
    if not os.path.exists(baseline_label_filename):
        TGS_Handler("E:/TGS/").create_dummy_baseline_labels_weekly(dataset)

        # create_dummy_baseline_labels_weekly(dataset)
    baseline_label_df = pd.read_csv(baseline_label_filename, header=None, names=['label'])
    baseline_labels = torch.from_numpy(np.array(baseline_label_df['label'].tolist())).to(args.device)
    return baseline_labels

class BaseLineModel:
    def __init__(self,data,dataset_name,test_ratio = 0.15):
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
    from script.utils.config import args, dataset_names
    from script.utils.data_util import loader

    datasets_list = pd.read_csv("../data/data_package/datasets_package_64.txt").iloc[:, 0].values

    result_path = '../data/output/baselinemodel.csv'

    for token in datasets_list:
        args.dataset = token
        data = loader(dataset=args.dataset, neg_sample=args.neg_sample)
        baseModel = BaseLineModel(data, args.dataset)
        auc, ap = baseModel.test()
        print(token,auc,ap)
        if not os.path.exists(result_path):
            result_df =pd.DataFrame(columns=["dataset", "auc", "ap"])
        else:
            result_df = pd.read_csv(result_path)
        result_df = result_df.append({'dataset': token,
                                               'auc': auc,
                                               'ap': ap,
                                               }, ignore_index=True)
        result_df.to_csv(result_path, index=False)