"""
Assumption:
    Train and test temporal graph classification task 
    without having a pre-trained models

April 2024
"""

import os
import math
import sys
import time
import torch
import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from script.configs.data_spec import DATA_PATH
# model_file_path = 'PUT MODEL PATH HERE'
# data_file_path = 'PUT RAW DATA PATH HERE'
model_file_path = f"{DATA_PATH}/output/ckpts/htgn/trainratio/"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
  

class MLP(torch.nn.Module):
  """
  Binary Classifier to work as a decoder

  reference:
    https://github.com/twitter-research/tgn/blob/master/utils/utils.py
  """
  def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, drop=0.5):
    super().__init__()
    self.fc_1 = torch.nn.Linear(in_dim, hidden_dim_1)
    self.fc_2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
    self.fc_3 = torch.nn.Linear(hidden_dim_1, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


def readout_function(embeddings, readout_scheme='mean'):
    """
    Read out function to generate a representation for the whole graph
    reference:    
    https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/readouts/basic_readout.py
    """
    # note: x.size(): [#nodes, args.n_out]
    if readout_scheme == 'max':
      readout = torch.max(embeddings, dim=0)[0].squeeze() # max readout
    elif readout_scheme == 'mean':
      readout = torch.mean(embeddings, dim=0).squeeze() # mean readout
    elif readout_scheme == 'sum':
      readout = torch.sum(embeddings, dim=0).squeeze() # sum readout
    else:
      readout = None
      raise ValueError("Readout Method Undefined!")
    return readout
  


def save_inference_results(model_name, mode, dataset, test_auc, test_ap, bias=False):
    result_folder = f"{model_file_path}/test_result/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if bias:
        result_path = "{}/{}_results_bias.csv".format(result_folder,model_name)
    else:
        result_path = "{}/{}_results.csv".format(result_folder,model_name)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "mode", "test_auc", "test_ap"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset, 'mode' : mode, 'test_auc': test_auc, 'test_ap': test_ap}, ignore_index=True)
    result_df.to_csv(result_path, index=False)


class Runner(object):
    def __init__(self):

        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr

        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]

        self.testLength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)]
        self.valLength = [math.floor(self.len[i] * args.val_ratio) for i in range(self.num_datasets)]

        self.train_shots = []
        self.val_shots = []
        self.test_shots = []
        self.teststart = []


        for i in range(self.num_datasets):
            test_start = self.len[i] - self.testLength[i]
            self.teststart.append(test_start)
            val_start = test_start - self.valLength[i]



            # Compute available room before validation set
            available_train_len = val_start
            desired_train_len = args.fixed_train_length

            if desired_train_len > available_train_len:
                train_start = 0
            else:
                train_start = val_start - desired_train_len

            train_range = list(range(train_start, val_start))
            val_range = list(range(val_start, test_start))
            test_range = list(range(test_start, self.len[i]))

            self.train_shots.append(train_range)
            self.val_shots.append(val_range)
            self.test_shots.append(test_range)
            print(f" Dataset = {i} , Len total = {self.len[i]} ,  test start = {test_start} , Len Test = {len(test_range)}, Test snapshots Min= {min(test_range)} , Test Snaphshot Max = {max(test_range)}")

        self.criterion = torch.nn.BCELoss()
        self.load_feature()

        # load the TG-models
        self.model = load_model(args).to(args.device)
        self.model_path = "{}/{}.pth".format(model_file_path, model_path)
        print("The models is going to be loaded from {}".format(self.model_path))
        # Load from check point
        #self.model.load_state_dict(torch.load(self.model_path)['model_state_dict'])
        self.model.load_state_dict(torch.load(self.model_path))
        # load the graph labels
        self.t_graph_labels, self.t_graph_feat, Node_Feat = multi_datasets_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4
        self.mlp_path = "{}/{}_mlp.pth".format(model_file_path, model_path)
        print("The MLP models is going to be loaded from {}".format(self.model_path))
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 
        # Load from check point
        # self.tgc_decoder.load_state_dict(torch.load(self.mlp_path)['model_state_dict'])
        self.tgc_decoder.load_state_dict(torch.load(self.mlp_path))

        self.optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                # self.x = torch.arange(args.max_node_id + 1).float().view(args.max_node_id + 1,1).to(args.device)
            args.nfeat = self.x.size(1)



    # def tgclassification_test(self, readout_scheme, dataset_idx):
    #     """
    #     Final inference on the test set
    #     """
    #     tg_labels, tg_preds, test_loss = [], [], []
    #     for t_test_idx, t in enumerate(self.test_shots[dataset_idx]):
    #         self.model.eval()
    #         self.tgc_decoder.eval()
    #         with torch.no_grad():
                
    #             edge_index = prepare(data[dataset_idx], t,args)[:3]
    #             embeddings = self.model(edge_index, self.x)

    #             # graph readout
    #             tg_readout = readout_function(embeddings, readout_scheme)
    #             tg_embedding = torch.cat((tg_readout,
    #                                       torch.from_numpy(self.t_graph_feat[dataset_idx][t_test_idx + 
    #                                                                                       len(self.train_shots[dataset_idx])+ 
    #                                                                                       len(self.val_shots[dataset_idx])]).to(args.device)))

    #             # graph classification
    #             tg_labels.append(self.t_graph_labels[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx]+ 
    #                                                                                       len(self.val_shots[dataset_idx]))].cpu().numpy())
    #             tg_preds.append(
    #                 self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
    #             self.model.update_hiddens_all_with(embeddings)
                
    #     auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
    #     return auc, ap
    def tgclassification_test(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds, test_loss = [], [], []
        for t_test_idx, t in enumerate(self.test_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index = prepare(data[dataset_idx], t, args)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = tg_readout
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][self.teststart[dataset_idx] + t_test_idx ]).to(args.device)))

                # graph classification
                tg_label = self.t_graph_labels[dataset_idx][self.teststart[dataset_idx] + t_test_idx ].float().view(1, )

                index = int((self.teststart[dataset_idx] + t_test_idx ))
                print(f"This is for dataset = {dataset_idx} ,  Label inndex Test = { index }")
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()
                
                tg_labels.append(tg_label.cpu().numpy())
                tg_preds.append(tg_pred.cpu().detach().numpy())
                self.model.update_hiddens_all_with(embeddings)
                test_loss.append(self.criterion(tg_pred, tg_label))
        tg_preds_binary = [1 if pred >= 0.5 else 0 for pred in tg_preds]
        # accuracy = accuracy_score(tg_labels, tg_preds_binary)
        # cf_matrix = confusion_matrix(tg_labels, tg_preds_binary)
        test_loss_tensor = torch.stack(test_loss)
        total_test_loss = (torch.mean(test_loss_tensor)).cpu().numpy()
        print(f"Test labels = {tg_labels}")
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

    def tgclassification_val(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_val_idx, t in enumerate(self.val_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index = prepare(data[dataset_idx], t,args)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = tg_readout
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_val_idx +
                                                                                          len(self.train_shots[dataset_idx])]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_val_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap


    def test(self):
        """
        Test the temporal graph classification task
        """
        
        self.model.init_hiddens()

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)     

        # Set model and decoder to evaluation mode   
        self.model.eval()
        self.tgc_decoder.eval()
        for dataset_idx in range(self.num_datasets):
            self.model.init_hiddens()
            data_name = args.dataset[dataset_idx]

            if args.test_bias:
                self.test_bias(dataset_idx)
            else:

                # Forwad pass through train data to get the embeddings
                for t_train in self.train_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index = prepare(data[dataset_idx], t_train,args)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                
                # Inference testing on validation set
                # val_auc, val_ap = self.tgclassification_val(self.readout_scheme, dataset_idx)
                # save_inference_results(model_path, 
                #              "Val", 
                #              data_name, 
                #              val_auc, 
                #              val_ap)
                
                # Forward pass through validation set to get the embeddings
                for t_val in self.val_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index = prepare(data[dataset_idx], t_val, args)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)

                # Inference testing on test set
                print(f"Testing Dataset {data_name}")
                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx)
                print(test_auc)
                save_inference_results(model_path, 
                             "Test", 
                             data_name, 
                             test_auc, 
                             test_ap)





if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import disease_path
    from script.nn.models.load_model import load_model
    from script.utils.data_util import load_multiple_datasets,prepare, multi_datasets_attributes_loading
    
    args.model = "HTGN"
    # args.seed = "710"
    pack = args.pack
    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))
    args.dataset, data = load_multiple_datasets("mix_dataset_package_test.txt")

    model_path = "{}_{}_seed_{}_fixed_train_length_{}_{}".format(args.model, pack, args.seed, args.fixed_train_length ,args.nout)
    runner = Runner()
    runner.test()
