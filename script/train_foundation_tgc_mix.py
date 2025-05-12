"""
Assumption:
    Train foundation temporal graph classification task 
    without having a pre-trained models
=======
April 2024

"""

import os
import math
import sys
import timeit
import torch
import numpy as np
import pandas as pd
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import wandb
import torch.nn.functional as F


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from script.configs.data_spec import DATA_PATH
# model_file_path = 'PUT MODEL PATH HERE'
# data_file_path = 'PUT RAW DATA PATH HERE'
model_file_path = f"{DATA_PATH}/output/ckpts/htgn/mix_models/"

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
        readout = torch.max(embeddings, dim=0)[0].squeeze()  # max readout
    elif readout_scheme == 'mean':
        readout = torch.mean(embeddings, dim=0).squeeze()  # mean readout
    elif readout_scheme == 'sum':
        readout = torch.sum(embeddings, dim=0).squeeze()  # sum readout
    else:
        readout = None
        raise ValueError("Readout Method Undefined!")
    return readout


def save_train_results(epoch, train_auc, train_ap, loss, time, dataset=None):
    """
    Saving Loss, AUC and AP for training per data and on average.
    If dataset is None, average over all datasets for each epoch is saved.
    """
    if dataset is None:
        result_folder = f"{model_file_path}/training_result/average"
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                              args.seed,
                                                                              len(args.dataset))
    else:
        result_folder = "{}/training_result/average/training_result/data/{}".format(model_file_path,dataset)
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                              args.seed,
                                                                              len(args.dataset))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)

    # Create a new DataFrame with the new data
    new_data = pd.DataFrame({
        'epoch': [int(epoch)],
        'train_loss': [loss],
        'train_auc': [train_auc],
        'train_ap': [train_ap],
        'time': [time]
    })

    # Use pd.concat instead of append
    result_df = pd.concat([result_df, new_data], ignore_index=True)

    # Save the updated DataFrame back to CSV
    result_df.to_csv(result_path, index=False)


def save_val_results(epoch, val_auc, val_ap, time, dataset=None):
    """
    Saving Validation AUC and Validation AP for each epoch, per data and average
    If dataset in None average is saved.
    """
    if dataset is None:
        result_folder = f"{model_file_path}/val_result/average"
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                              args.seed,
                                                                              len(args.dataset))
    else:
        result_folder = "{}/val_result/average/val_result/data/{}".format(model_file_path, dataset)
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                              args.seed,
                                                                              len(args.dataset))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)

    # Create a new DataFrame with the new data
    new_data = pd.DataFrame({'epoch': [int(epoch)], 'test_auc': [val_auc], 'test_ap': [val_ap], 'time': [time]})

    # Use pd.concat instead of append
    result_df = pd.concat([result_df, new_data], ignore_index=True)

    # Save the updated DataFrame back to CSV
    result_df.to_csv(result_path, index=False)


class Runner(object):
    def __init__(self):
        # Initialize Wandb for this project and run
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ScalingTGNs",
                name="{}_{}_{}".format(args.model, args.seed, len(data)),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,
                }
            )

        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr
        self.start_epoch = 0

        # Calculate the length of validation and test sets and split the datasets individually
        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        self.testLength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)]
        self.valLength = [math.floor(self.len[i] * args.val_ratio) for i in range(self.num_datasets)]

        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - self.testLength[i] - self.valLength[i])) for i in
                            range(self.num_datasets)]
        self.val_shots = [
            list(range(self.len[i] - self.testLength[i] - self.valLength[i], self.len[i] - self.testLength[i])) for i in
            range(self.num_datasets)]
        self.test_shots = [list(range(self.len[i] - self.testLength[i], self.len[i])) for i in range(self.num_datasets)]

        # Use Binary Cross Entropy for calculatin loss
        self.criterion = torch.nn.BCELoss()

        # Loading graph features
        self.load_feature()

        # Loading model
        self.model = load_model(args).to(args.device)

        # Paths for saving model and checkpoints
        self.model_path = '{}/{}_{}_seed_{}'.format(model_file_path,
                                                                    args.model,
                                                                    self.num_datasets,
                                                                    args.seed)

        self.model_chkp_path = '{}/{}_{}_seed_{}'.format(model_file_path,
                                                                                    args.model,
                                                                                    self.num_datasets,
                                                                                    args.seed)

        # load the graph-pooled features and graph labels
        self.t_graph_labels, self.t_graph_feat, TG_node_feats_data = multi_datasets_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout + num_extra_feat, hidden_dim_1=args.nout + num_extra_feat,
                               hidden_dim_2=args.nout + num_extra_feat,
                               drop=0.1)  # @NOTE: these hyperparameters may need to be changed

        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr)

        # If checkpoint of current run is saved, load and resume training 
        if os.path.exists("{}.pth".format(self.model_chkp_path)):
            logger.info("INFO: Model already exist and will be loaded from {}".format(self.model_chkp_path))
            checkpoint = torch.load("{}.pth".format(self.model_chkp_path))
            self.model.load_state_dict(checkpoint['model_state_dict'])

            mlp_checkpoint = torch.load("{}_mlp.pth".format(self.model_chkp_path))
            self.tgc_decoder.load_state_dict(mlp_checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']
            logger.info("INFO: Epochs will start from: {}".format(self.start_epoch))

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
                # self.x = np.fill_diagonal(torch.zeros(args.num_nodes, args.num_nodes), 
                #                           args.node_ids).to(args.device)
                logger.info('INFO: using one-hot feature')
            # Comment when generating cache
            args.nfeat = self.x.size(1)

    def tgclassification_val(self, epoch, readout_scheme, dataset_idx):
        """
        Final inference on the validation set
        """
        tg_labels, tg_preds = [], []

        for t in self.val_shots[dataset_idx]:
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_index = prepare(data[dataset_idx], t, args)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = tg_readout
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def tgclassification_test(self, epoch, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t in self.test_shots[dataset_idx]:
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_index = prepare(data[dataset_idx], t, args)[:3]
                embeddings = self.model(edge_index, list(self.x))

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = tg_readout
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def run(self):
        """
        Run the temporal graph classification task
        """
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        # Set the model and decoder to train mode
        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()

        t_total_start = timeit.default_timer()
        best_model = self.model.state_dict()
        patience = 0
        best_val_auc = -1

        for epoch in range(self.start_epoch + 1, args.max_epoch + 1):
            print("Epoch: ", epoch)
            t_epoch_start = timeit.default_timer()
            epoch_losses = []  # Saving average losses of datasets in each epoch
            train_aucs, train_aps, val_aucs, val_aps = [], [], [], []
            test_aucs, test_aps = [], []

            # Shuffling order of datasets for each epoch
            dataset_rnd = random.sample(range(self.num_datasets), self.num_datasets)

            for dataset_idx in dataset_rnd:
                tg_labels, tg_preds = [], []
                # data_name = args.data_name[args.dataset[dataset_idx]] if args.dataset[dataset_idx] in args.data_name else args.dataset[dataset_idx]
                self.model.train()
                self.tgc_decoder.train()
                self.model.init_hiddens()
                dataset_losses = []  # Store losses for each dataset in one epoch

                for t_train in self.train_shots[dataset_idx]:
                    print(f"Training for snapshot : {t_train}/{len(self.train_shots[dataset_idx])}")
                    self.optimizer.zero_grad()
                    edge_index = prepare(data[dataset_idx], t_train, args)
                    embeddings = self.model(edge_index, self.x)

                    # graph readout
                    tg_readout = readout_function(embeddings, self.readout_scheme)
                    tg_embedding = tg_readout
                    tg_embedding = torch.cat((tg_readout,
                                              torch.from_numpy(self.t_graph_feat[dataset_idx][t_train]).to(
                                                  args.device)))

                    # graph classification
                    tg_label = self.t_graph_labels[dataset_idx][t_train].float().view(1, )
                    tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                    tg_labels.append(tg_label.cpu().numpy())
                    tg_preds.append(tg_pred.cpu().detach().numpy())
                    train_loss = self.criterion(tg_pred, tg_label)
                    train_loss.backward()
                    self.optimizer.step()
                    dataset_losses.append(train_loss.item())
                    # update the models
                    self.model.update_hiddens_all_with(embeddings)

                if isnan(train_loss):
                    print('ATTENTION: nan loss')
                    break

                # --------------------Evaluation------------------------
                # Foundational model evaluation:
                print("evaluating")
                avg_dataset_loss = np.mean(dataset_losses)
                epoch_losses.append(avg_dataset_loss)

                self.model.eval()
                self.tgc_decoder.eval()
                train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
                val_epoch, val_auc, val_ap = self.tgclassification_val(epoch, self.readout_scheme, dataset_idx)
                train_aucs.append(train_auc)
                train_aps.append(train_ap)
                val_aucs.append(val_auc)
                val_aps.append(val_ap)
                # Save train and validation results for each dataset
                print("saving")
                save_val_results(epoch, val_auc, val_ap, 0, dataset=args.dataset[dataset_idx])
                save_train_results(epoch, train_auc, train_ap, avg_dataset_loss, 0, dataset=args.dataset[dataset_idx])

                # Updating wandb for each dataset
                if (args.wandb):
                    wandb.log({"Data {} Train Loss".format(args.dataset[dataset_idx]): avg_dataset_loss,
                               "Data {} Train AUC".format(args.dataset[dataset_idx]): train_auc,
                               "Data {} Train AP".format(args.dataset[dataset_idx]): train_ap,
                               "Data {} Val AUC".format(args.dataset[dataset_idx]): val_auc,
                               "Data {} Val AP".format(args.dataset[dataset_idx]): val_ap,
                               "Epoch": epoch
                               })

            # Calculte average of Loss, AUC and AP for train and validations among datasets for each epoch
            avg_epoch_loss = np.mean(epoch_losses)
            avg_train_auc = np.mean(train_aucs)
            avg_train_ap = np.mean(train_aps)
            avg_val_auc = np.mean(val_aucs)
            avg_val_ap = np.mean(val_aps)
            total_epoch_time = timeit.default_timer() - t_epoch_start

            # Saving model checkpoint:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()},
                       "{}.pth".format(self.model_chkp_path))

            torch.save({'epoch': epoch,
                        'model_state_dict': self.tgc_decoder.state_dict()},
                       "{}_mlp.pth".format(self.model_chkp_path))

            # Saving best model based on validation AUC
            if best_val_auc < avg_val_auc or epoch <= args.min_epoch:
                patience = 0
                best_val_auc = avg_val_auc
                best_model = self.model.state_dict()  # Saved the best model for testing
                best_mlp = self.tgc_decoder.state_dict()
                best_test_results = [test_aucs, test_aps]
                best_epoch = epoch
            else:  # Check for early stopping
                if best_val_auc - np.mean(val_aucs) > 0.005:
                    patience += 1
                    print("patience at epoch {}: {}".format(epoch, patience))
                if patience > args.patience:
                    logger.info("Early stopping at epoch: {}".format(epoch))
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          total_epoch_time,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}, Loss: {:.4f}".format(epoch, avg_train_auc, avg_train_ap,
                                                                                    avg_epoch_loss))
                logger.info(
                    "Val: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(val_epoch, avg_val_auc, avg_val_ap))

            if (args.wandb):
                wandb.log({"Avg Train Loss": avg_epoch_loss,
                           "Avg Train AUC": avg_train_auc,
                           "Avg Train AP": avg_train_ap,
                           "Avg Val AUC": avg_val_auc,
                           "Avg Val AP": avg_val_ap,
                           "Epoch": epoch
                           })

            # Save average train and validation results for each epoch
            save_val_results(epoch, avg_val_auc, avg_val_ap, total_epoch_time)
            save_train_results(epoch, avg_train_auc, avg_train_ap, avg_epoch_loss, total_epoch_time)

        total_time = timeit.default_timer() - t_total_start
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # Saving best model and decoder
        logger.info("INFO: Saving best model from epoch {} ...".format(best_epoch))
        logger.info("File name: {}_seed_{}_{}.pth".format(args.model, args.seed, self.num_datasets))
        torch.save(best_model, "{}_{}.pth".format(self.model_path, args.nout))
        torch.save(best_mlp, "{}_{}_mlp.pth".format(self.model_path, args.nout))
        logger.info("Best test results: {}".format(best_test_results))


if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.nn.models.load_model import load_model
    from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading

    args.model = "HTGN"
    # args.seed = 710
    args.max_epoch = 200
    args.lr = 0.0001
    args.log_interval = 10
    args.patience = 30


    # args.wandb = True
    print("INFO: >>> Temporal Graph Classification <<<")
    print("======================================")
    print("INFO: Model: {}".format(args.model))

    # Load the datasets listed on the .txt file
    args.dataset, data = load_multiple_datasets("mix_dataset_package_{}.txt".format(args.pack))

    init_logger(
        '../../data/output/log/{}_{}_seed_{}_log.txt'.format(args.model,  len(args.dataset), args.seed))
    set_random(args.seed)
    logger.info("INFO: Number of data: {}, seed: {}".format(len(data), args.seed))

    runner = Runner()
    runner.run()
