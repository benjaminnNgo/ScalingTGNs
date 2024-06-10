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
    partial_path = f'../data/input/raw/dummy_labels'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    # load baseline model labels
    baseline_label_filename = f'{partial_path}/{dataset}_dummy_fd_ld_labels.csv'
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
    from script.utils.config import args
    from script.utils.data_util import loader
    # csv_lists = [
    #     "unnamedtoken216540x09a3ecafa817268f77be1283176b946c4ff2e608",
    #     "unnamedtoken223250xf2ec4a773ef90c58d98ea734c0ebdb538519b988",
    #     "unnamedtoken222800xa49d7499271ae71cd8ab9ac515e6694c755d400c",
    #     "unnamedtoken223030x4ad434b8cdc3aa5ac97932d6bd18b5d313ab0f6f",
    #     "unnamedtoken220850x9fa69536d1cda4a04cfb50688294de75b505a9ae",
    #     "unnamedtoken220220xade00c28244d5ce17d72e40330b1c318cd12b7c3",
    #     "unnamedtoken223090xc4ee0aa2d993ca7c9263ecfa26c6f7e13009d2b6",
    #     "unnamedtoken221090x5de8ab7e27f6e7a1fff3e5b337584aa43961beef",
    #     "unnamedtoken220240x235c8ee913d93c68d2902a8e0b5a643755705726",
    #     "unnamedtoken221150xa2cd3d43c775978a96bdbf12d733d5a1ed94fb18",
    #     "unnamedtoken218340xaa6e8127831c9de45ae56bb1b0d4d4da6e5665bd",
    #     "unnamedtoken220960x4da27a545c0c5b758a6ba100e3a049001de870f5",
    #     "unnamedtoken217780x7dd9c5cba05e151c895fde1cf355c9a1d5da6429",
    #     "unnamedtoken220250xa71d0588eaf47f12b13cf8ec750430d21df04974",
    #     "unnamedtoken218270x5026f006b85729a8b14553fae6af249ad16c9aab",
    #     "unnamedtoken221900x49642110b712c1fd7261bc074105e9e44676c68f",
    #     "unnamedtoken216900x9e32b13ce7f2e80a01932b42553652e053d6ed8e",
    #     "unnamedtoken218450x221657776846890989a759ba2973e427dff5c9bb",
    #     "TRAC0xaa7a9ca87d3694b5755f213b5d04094b8d0f0a6f",
    #     "unnamedtoken220280xcf3c8be2e2c42331da80ef210e9b1b307c03d36a"
    # ]
    csv_lists = [
    "unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0",
    "unnamedtoken216240x83e6f1e41cdd28eaceb20cb649155049fac3d5aa",
    "unnamedtoken216300xcc4304a31d09258b0029ea7fe63d032f52e44efe",
    "unnamedtoken216350xe53ec727dbdeb9e2d5456c3be40cff031ab40a55",
    "unnamedtoken216360xfca59cd816ab1ead66534d82bc21e7515ce441cf",
    "unnamedtoken216390x1ceb5cb57c4d4e2b2433641b95dd330a33185a44",
    "unnamedtoken216540x09a3ecafa817268f77be1283176b946c4ff2e608",
    "unnamedtoken216550xbcca60bb61934080951369a648fb03df4f96263c",
    "unnamedtoken216580x5f98805a4e8be255a32880fdec7f6728c6568ba0",
    "unnamedtoken216620x429881672b9ae42b8eba0e26cd9c73711b891ca5",
    "unnamedtoken214030x07e0edf8ce600fb51d44f51e3348d77d67f298ae",
    "CMT0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f",
    "unnamedtoken218230x320623b8e4ff03373931769a31fc52a4e78b5d70",
    "unnamedtoken169170x06325440d014e39736583c165c2963ba99faf14e"
    ]

    result_path = '../data/output/baselinemodel.csv'
    # if not os.path.exists(result_path):
    #     result_df = pd.DataFrame(columns=['dataset','auc','ap'])
    # else:
    #     result_df = pd.read_csv(result_path)
    #
    # for csv_file_name in csv_lists:
    #     try:
    #         dataname = csv_file_name.replace("_","").replace(".csv","")
    #         data = loader(dataset=dataname, neg_sample="rnd")
    #         # createDummyBaselineLabelsWeekly(csv_file_name)
    #         baseModel = BaseLineModel(data,dataname)
    #         auc, ap = baseModel.test()
    #
    #
    #         result_df = result_df._append({'dataset': dataname,
    #                                        'auc': auc,
    #                                        'ap': ap,
    #                                        }, ignore_index=True)
    #         result_df.to_csv(result_path, index=False)
    #     except Exception as e:
    #         print("Can't process csv file {}".format(csv_file_name))
    for token in csv_lists:
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
