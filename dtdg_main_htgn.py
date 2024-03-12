import os
import sys
import timeit
import torch
import numpy as np
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
import wandb

"""
#! script for running discrete model on DTDG datasets
all training, val and test splits are in snapshots format
also need to generate negative samples with integer timestamps
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def generate_random_negatives(pos_edges: torch.Tensor,
                              num_nodes: int, 
                              num_neg_samples: int =1):
    r"""
    generate random negative samples for training
    Parameters:
        pos_edges: positive edges
        num_nodes: number of nodes in the graph
        num_neg_samples: number of negative samples per positive edge
    Return:
        neg_edges: negative edges
    """
    neg_edges = negative_sampling(pos_edges, num_nodes=num_nodes, num_neg_samples=(pos_edges.size(1)*num_neg_samples))
    return neg_edges



class Runner(object):
    def __init__(self):
        
        self.train_data = data['train_data']
        self.val_data = data['val_data']
        self.test_data = data['test_data']
        args.num_nodes = data['train_data']['num_nodes'] + 1

        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="utg",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                "time granularity": args.time_scale,
                }
            )


        # set initial features; the assumption is to have trainable features
        self.x = None  # TODO: @Andy --> this might need to change if we want to load initial features
        
        # set the model
        self.model = load_model(args).to(args.device)

        
        # set the loss
        self.loss = ReconLoss(args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
        
    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer
    

    #! assumes there is no time gap, throws error if so 
    def test(self, 
            test_data,
            neg_sampler,
            evaluator,  
            embeddings,
            metric,
            split_mode):
        r"""
        test on discrete dataset snapshots, go through edges and load ns samples after entire snapshot is tested, forward pass only to update node embeddings 
        Parameters:
            test_snapshots: the test set in the form of snapshots
            neg_sampler: TGB negative sampler
            evaluator: TGB evaluator
            embeddings: the embeddings of the model
            metric: the metric to be used for evaluation
            split_mode: the split mode for the negative sampler
        """
        embeddings = embeddings.detach()
        test_snapshots = test_data['edge_index'] #converted to undirected, also removes self loops as required by HTGN
        test_edges = test_data['original_edges'] #original edges unmodified
        ts_min = min(test_snapshots.keys())

        perf_list = {}
        perf_idx = 0

        for snapshot_idx in test_snapshots.keys():
            pos_index = torch.from_numpy(test_edges[snapshot_idx])
            pos_index = pos_index.long().to(args.device)

            for i in range(pos_index.shape[0]):
                pos_src = pos_index[i][0].item()
                pos_dst = pos_index[i][1].item()
                pos_t = snapshot_idx
                neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode=split_mode)
                
                for idx, neg_batch in enumerate(neg_batch_list):
                    query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                    query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                    query_src = torch.from_numpy(query_src).long().to(args.device)
                    query_dst = torch.from_numpy(query_dst).long().to(args.device)
                    edge_index = torch.stack((query_src, query_dst), dim=0)
                    y_pred = self.loss.predict_link(embeddings, edge_index)

                    input_dict = {
                            "y_pred_pos": np.array([y_pred[0]]),
                            "y_pred_neg": np.array(y_pred[1:]),
                            "eval_metric": [metric],
                        }
                    perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                    perf_idx += 1

            #* update the node embeddings with edges after the snapshot has been predicted
            prev_index = test_snapshots[snapshot_idx]
            prev_index = prev_index.long().to(args.device)
            z = self.model(prev_index, self.x)
            embeddings = self.model.update_hiddens_all_with(z)

        result = list(perf_list.values())
        perf_list = np.array(result)
        perf_metrics = float(np.mean(perf_list))

        return perf_metrics, embeddings

    def run(self, seed=1):
        set_random(seed)
        optimizer = self.optimizer()  # @TODO: RiemannianAdam or Adam?!
        self.model.train()

        best_val = 0 
        best_test = 0
        best_epoch = 0

        for epoch in range(1, args.max_epoch + 1):
            epoch_start_time = timeit.default_timer()
            
            epoch_losses = []
            self.model.init_hiddens()
            # ==========================
            # Train
            # important to reset the node embeddings z at the start of the epoch
            train_start_time = timeit.default_timer()
            self.model.train()
            z = None
            snapshot_list = self.train_data['edge_index']
            """
            #! significant modification to the training procedure in order to receive test time updates
            1. feed the true edges from (t-1) into the model and update the embeddings
            2. use the updated embeddings to predict the edges in t
            exception is for the first snapshot
            GNN generates embeddings based on true edges not negative samples
            """

            for snapshot_idx in range(self.train_data['time_length']):
                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)
                optimizer.zero_grad()

                #* generate random samples for training
                neg_index = generate_random_negatives(pos_index, num_nodes=args.num_nodes, num_neg_samples=1)

                """
                #! here only positive edges should pass through the model
                in original code, edge index only comes from positive edges thus neg index is not used
                only the z are used for prediction
                """
                #edge_index = torch.cat((pos_index, neg_index), dim=1)
                if (snapshot_idx == 0):
                    edge_index = pos_index
                    z = self.model(edge_index, self.x)
                               

                if args.use_htc == 0:
                    epoch_loss = self.loss(z, pos_edge_index=edge_index,  neg_edge_index=neg_index)
                else:
                    epoch_loss = self.loss(z, pos_edge_index=edge_index, neg_edge_index=neg_index) + self.model.htc(z)
                
                epoch_loss.backward()
                optimizer.step()
                epoch_losses.append(epoch_loss.item())

                #* update the embedding after the prediction of current snapshot
                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)
                z = self.model(pos_index, self.x)
                self.model.update_hiddens_all_with(z)
            
            average_epoch_loss = np.mean(epoch_losses)
            train_end_time = timeit.default_timer()

            z = z.detach()

            self.model.eval()

            evaluator = Evaluator(name="tgbl-wiki") #reuse MRR evaluator from TGB
            metric = "mrr"
            neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

            #* load the val negative samples
            neg_sampler.load_eval_set(fname=args.dataset + "_val_ns.pkl", split_mode="val",)

            #* steps for snapshots edges in test set
            #1. discretize the test set into snapshots
            #2. map from integer to unix ts
            val_start_time = timeit.default_timer()
            val_metrics, z = self.test(self.val_data, neg_sampler, evaluator, z, metric, 'val')
            val_end_time = timeit.default_timer()


            epoch_end_time = timeit.default_timer()
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            
            logger.info("Epoch:{}, Time (train + val): {:.4f}, GPU Mem.: {:.1f}MiB".format(epoch,
                                                                                epoch_end_time - epoch_start_time,
                                                                                gpu_mem_alloc))
            logger.info("\tTrain: Loss: {:.4f}, Elapsed time: {:.4f}".format(average_epoch_loss, train_end_time - train_start_time))
            logger.info("\tValidation: {}: {:.4f}, Elapsed time: {:.4f}".format(metric, val_metrics, val_end_time - val_start_time))

            if (args.wandb):
                wandb.log({"train_loss": average_epoch_loss,
                        "val_" + metric: val_metrics,
                        "train time": train_end_time - train_start_time,
                        "val time": val_end_time - val_start_time,
                        })
                
                
            #! report test results when validation improves
            if (val_metrics > best_val):
                best_val = val_metrics
                neg_sampler.load_eval_set(fname=args.dataset + "_test_ns.pkl", split_mode="test",)

                test_start_time = timeit.default_timer()
                test_metrics, z = self.test(self.test_data, neg_sampler, evaluator, z, metric, 'test')
                test_end_time = timeit.default_timer()

                logger.info("\tTest: {}: {:.4f}, Elapsed time: {:.4f}".format(metric, test_metrics, test_end_time - test_start_time))

                best_test = test_metrics
                #* implementing patience
                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    print ("run finishes")
                    print ("best epoch is, ", best_epoch)
                    print ("best val performance is, ", best_val)
                    print ("best test performance is, ", best_test)
                    return

                best_epoch = epoch
            
        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        print ("best val performance is, ", best_val)
        print ("best test performance is, ", best_test)



if __name__ == '__main__':
    from utils.configs import args
    from utils.log_utils import logger, init_logger
    from utils.utils_func import set_random
    from models.load_model import load_model
    from models.loss import ReconLoss, VGAEloss
    from utils.data_util import loader, prepare_dir

    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    init_logger(prepare_dir(args.output_folder) + args.dataset + '_timeScale_' + str(args.time_scale) + '_seed_' + str(args.seed) + '.log')

    runner = Runner()
    for seed in range(args.seed, args.seed + args.num_runs):
        print ("--------------------------------")
        print ("excuting run with seed ", seed)
        runner.run(seed=seed)
        print ("--------------------------------")
