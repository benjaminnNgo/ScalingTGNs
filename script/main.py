import os
import sys
import time
import torch
import numpy as np
from math import isnan
import wandb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="scalingTGNs",
                #Set name of the run:
                name="{}_{}".format(args.dataset, args.model),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        self.load_feature()
        self.model = load_model(args).to(args.device)
        self.model_path = '../saved_models/{}/{}_{}_seed_{}.pth'.format(args.dataset, args.dataset,
                                                                   args.model, args.seed)
        print("INFO: models is going to be saved at {}".format(self.model_path))
        
        self.loss = ReconLoss(args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
        print("INFO: Args: ", args)
        logger.info('INFO: total length: {}, test length: {}'.format(self.len, args.testlength))

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("INFO: using trainable feature, feature dim: {}".format(args.nfeat))
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

    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def run(self):
        optimizer = self.optimizer()  # @TODO: should I use Adam to be consistent with other baselines?!
        t_total0 = time.time()#=======================fix this
        test_results, min_loss = [0] * 5, 10
        self.model.train()
        for epoch in range(1, 21):
        # for epoch in range(1, 2):
            t0 = time.time()
            epoch_losses = []
            self.model.init_hiddens()
            # train
            self.model.train()
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t)
                optimizer.zero_grad()
                z = self.model(edge_index, self.x)
                if args.use_htc == 0:
                    # epoch_loss = self.loss(z, edge_index)  # this was default!!! It doesn't make sense to me!!!
                    epoch_loss = self.loss(z, pos_index, neg_index)
                else:
                    # epoch_loss = self.loss(z, edge_index) + self.models.htc(z)  # so as this one!
                    epoch_loss = self.loss(z, pos_index, neg_index) + self.model.htc(z)
                epoch_loss.backward()
                optimizer.step()
                epoch_losses.append(epoch_loss.item())
                self.model.update_hiddens_all_with(z)
            self.model.eval()
            average_epoch_loss = np.mean(epoch_losses)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                test_results = self.test(epoch, z)
                patience = 0
            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    print('INFO: Early Stopping...')
                    break
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("INFO: Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                          time.time() - t0,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Epoch:{:}, Test AUC: {:.4f}, AP: {:.4f}, New AUC: {:.4f}, New AP: {:.4f}".format(test_results[0],
                                                                                                      test_results[1],
                                                                                                      test_results[2],
                                                                                                      test_results[3],
                                                                                                      test_results[4]))


            if isnan(epoch_loss):
                print('nan loss')
                break
            if (args.wandb):
                wandb.log({"train_loss": average_epoch_loss,
                           "test_AUC": test_results[1],
                           "AP": test_results[2],
                           "New AUC": test_results[3],
                           "New AP": test_results[4]

                           })
        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # saving the trained models
        logger.info("INFO: Saving the models...")
        torch.save(self.model.state_dict(), self.model_path)
        logger.info("INFO: The models is saved. Done.")



    def test(self, epoch, embeddings=None):
        auc_list, ap_list = [], []
        auc_new_list, ap_new_list = [], []
        embeddings = embeddings.detach()
        for t in self.test_shots:
            edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
            new_pos_edge, new_neg_edge = prepare(data, t)[-2:]
            auc, ap = self.loss.predict(embeddings, pos_edge, neg_edge)
            auc_new, ap_new = self.loss.predict(embeddings, new_pos_edge, new_neg_edge)
            auc_list.append(auc)
            ap_list.append(ap)
            auc_new_list.append(auc_new)
            ap_new_list.append(ap_new)
        if epoch % args.log_interval == 0:
            logger.info(
                'Epoch:{} (test_method); Transductive: average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_list), np.mean(ap_list)))
            logger.info('Epoch:{} (test_method); Inductive: average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_new_list),
                                                                                   np.mean(ap_new_list)))
        return epoch, np.mean(auc_list), np.mean(ap_list), np.mean(auc_new_list), np.mean(ap_new_list)



if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir, load_multiple_datasets
    from script.inits import prepare

    print("INFO: Dataset: {}".format(args.dataset))
    data = loader(dataset=args.dataset, neg_sample=args.neg_sample)
    args.num_nodes = data['num_nodes']
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.dataset + '_seed_' + str(args.seed) + '.txt')
    runner = Runner()
    runner.run()
