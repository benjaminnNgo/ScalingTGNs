"""
Assumption:
    Train and test temporal graph classification task 
    without having a pre-trained models

July 14, 2023
"""

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
# wandb.login("29968c684c2e412ed650ce0b5b52db584d572b86")

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
  

def extra_dataset_attributes_loading(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/Baseline'
    TG_labels_data = []
    TG_feats_data = []
    for dataset in args.dataset:
    # load graph lables
        label_filename = f'{partial_path}/Labels/{dataset}'
        label_df = pd.read_csv(label_filename, header=None, names=['label'])
        TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)
        TG_labels_data.append(TG_labels)
        
        # load and process graph-pooled (node-level) features 
        edgelist_filename = f'{partial_path}/{dataset}'
        edgelist_df = pd.read_csv(edgelist_filename)
        uniq_ts_list = np.unique(edgelist_df['snapshot'])
        TG_feats = []
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['from', 'to', 'value']]
            ts_G = nx.from_pandas_edgelist(ts_edges, source='from', target='to', edge_attr='value', create_using=nx.MultiDiGraph)
            node_list = list(ts_G.nodes)
            indegree_list = np.array(ts_G.in_degree(node_list))
            weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='value'))
            outdegree_list = np.array(ts_G.out_degree(node_list))
            weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='value'))

            if readout_scheme == 'max':
                TG_this_ts_feat = np.array([np.max(indegree_list), np.max(weighted_indegree_list), 
                                            np.max(outdegree_list), np.max(weighted_outdegree_list)])
            elif readout_scheme == 'mean':
                TG_this_ts_feat = np.array([np.mean(indegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_indegree_list[: , 1].astype(float)), 
                                            np.mean(outdegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_outdegree_list[: , 1].astype(float))])
            elif readout_scheme == 'sum':
                TG_this_ts_feat = np.array([np.sum(indegree_list), np.sum(weighted_indegree_list), 
                                            np.sum(outdegree_list), np.sum(weighted_outdegree_list)])
            else:
                TG_this_ts_feat = None
                raise ValueError("Readout scheme is Undefined!")
            
            TG_feats.append(TG_this_ts_feat)
    
        # scale the temporal graph features to have a reasonable range
        scalar = MinMaxScaler()
        TG_feats = scalar.fit_transform(TG_feats)
        TG_feats_data.append(TG_feats)

    return TG_labels_data, TG_feats_data
  

class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ScalingTGNs",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                }
            )

        self.readout_scheme = 'mean'
        self.tgc_lr = 1e-4

        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        # self.len = data['time_length']
        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - args.testlength)) for i in range(self.num_datasets)] #Changed
        self.test_shots = [list(range(self.len[i] - args.testlength, self.len[i])) for i in range(self.num_datasets)] #Changed
        args.num_nodes = max(args.num_nodes)
        
        # self.train_shots = list(range(self.start_train, self.len - args.testlength))
        # self.test_shots = list(range(self.len - args.testlength, self.len))
        self.load_feature()
        logger.info('INFO: total length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))

        self.model = load_model(args).to(args.device)
        self.model_path = '../saved_models/fm/'
        # logger.info("The models is going to be loaded from {}".format(self.model_path))
        # self.models.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("INFO: Using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('INFO: using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('INFO: using one-hot feature')
            args.nfeat = self.x.size(1)


    def tgclassification_test(self, epoch, readout_scheme):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_test_idx, t in enumerate(self.test_shots):
           self.model.eval()
           self.tgc_decoder.eval()
           with torch.no_grad():
              edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
              new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

              embeddings = self.model(edge_index, self.x)

              # graph readout
              tg_readout = readout_function(embeddings, readout_scheme)
              tg_embedding = torch.cat((tg_readout, 
                                        torch.from_numpy(self.t_graph_feat[t_test_idx + len(self.train_shots)]).to(args.device)))
          
              # graph classification
              tg_labels.append(self.t_graph_labels[t_test_idx + len(self.train_shots)].cpu().numpy())
              tg_preds.append(self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap
        

    def run(self):
        """
        Run the temporal graph classification task
        """
        # define optimizer and criterion
        optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )
        criterion = torch.nn.BCELoss()

        # load the TG-models
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
        
        t_total_start = time.time()
        min_loss = 10
        dataset_loss = []
        for dataset_idx in range(self.num_datasets): 
            self.model.init_hiddens()
            self.model.train()
            train_avg_epoch_loss_dict = {}
            
            for epoch in range(1, args.max_epoch + 1):
                t_epoch_start = time.time()
                epoch_losses = []
                for t_train_idx, t_train in enumerate(self.train_shots[dataset_idx]):
                    optimizer.zero_grad()

                    edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data[dataset_idx], t_train)
                    embeddings = self.model(edge_index, self.x)
                    
                    # graph readout
                    tg_readout = readout_function(embeddings, self.readout_scheme)
                    tg_embedding = torch.cat((tg_readout, 
                                              torch.from_numpy(self.t_graph_feat[t_train_idx + len(self.train_shots[dataset_idx])]).to(args.device)))
                    
                    # graph classification
                    tg_label = self.t_graph_labels[t_train_idx].float().view(1, )
                    tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                    t_loss = criterion(tg_pred, tg_label)
                    t_loss.backward()
                    optimizer.step()
                    epoch_losses.append(t_loss.item())
                    # update the models
                    self.model.update_hiddens_all_with(embeddings)

                # --------------------Evaluation------------------------
                self.model.eval()
                avg_epoch_loss = np.mean(epoch_losses)
                train_avg_epoch_loss_dict[epoch] = avg_epoch_loss

                # patience = 0  I don't think it should be zero here
                if avg_epoch_loss < min_loss:
                        min_loss = avg_epoch_loss
                        test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, self.readout_scheme)
                        patience = 0
                else:
                        patience += 1
                        if epoch > args.min_epoch and patience > args.patience:  # NOTE: args.min_epoch prevents it from stopping early in most cases
                            print('INFO: Early Stopping...')
                            break
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

                if epoch == 1 or epoch % args.log_interval == 0:
                        logger.info('==' * 30)
                        logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                                time.time() - t_epoch_start,
                                                                                                gpu_mem_alloc))
                        logger.info(
                            "Test: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))
                
                if isnan(t_loss):
                        print('ATTENTION: nan loss')
                        break
                
                if (args.wandb):
                    wandb.log({"Train Loss": avg_epoch_loss,
                               "Test Loss" : test_epoch,
                               "Test AUC" : test_auc,
                               "Test AP" : test_ap,
                        })
            dataset_loss.append(avg_epoch_loss)
            # logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
            # logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

            # saving the trained models
            logger.info("INFO: Saving the models...")
            curr_stage_model_path = (self.model_path + '{}_{}_seed_{}.pth'.format(dataset_idx,
                                                                   args.models, args.seed))
            torch.save(self.model.state_dict(), curr_stage_model_path)
            logger.info("INFO: The models is saved. Done.")


        logger.info('>> Total time : %6.2f' % (time.time() - t_total_start))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # ------------ DEBUGGING ------------
        # save the training loss values
        # partial_results_path = f'../data/output/log/{args.dataset}/{args.model}/'
        # loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        # with open(loss_log_filename, 'wb') as file:
        #     dump(train_avg_epoch_loss_dict, file)
        
        # plotting the training losses
        # train_avg_epoch_loss_dict = load(open(loss_log_filename, 'rb'))
        # train_values = train_avg_epoch_loss_dict.values()
        # epoch_range = range(0, epoch)
        # plt.plot(epoch_range, train_values, label='Training Loss')
        # plt.title('Training Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(np.arange(0, epoch, 50))
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.png')
        # -----------------------------------
        # -----------------------------------

        # Final Test
        # test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, self.readout_scheme)
        # logger.info("Final Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))


if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir, load_multiple_datasets
    from script.inits import prepare

    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))
    args.dataset = ["unnamed_token_21655_0xbcca60bb61934080951369a648fb03df4f96263c.csv",
                 "unnamed_token_21662_0x429881672b9ae42b8eba0e26cd9c73711b891ca5.csv"]
    data = load_multiple_datasets(args.dataset, args.neg_sample)
    args.num_nodes = [data[i]['num_nodes'] for i in range(len(data))]
    # data = loader(dataset=args.dataset, neg_sample=args.neg_sample)
    # args.num_nodes = data['num_nodes']
    print("INFO: Number of nodes:", args.num_nodes)
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.model + '_' + "Test1" + '_seed_' + str(args.seed) + '_log.txt')
    runner = Runner()
    runner.run()


# ----------------------
# commands to run:
# cd script
# python train_tgc_end_to_end.py --models=HTGN --seed=710  --dataset=dgd --max_epoch=200