import os
import sys
import time
import timeit
import torch
import numpy as np
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import wandb


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
        
        #* for tgb dataset, we will convert the val and test set on the fly here instead of loading. 
        self.val_data = data['val_data']
        self.test_data = data['test_data']

        args.num_nodes = data['train_data']['num_nodes'] + int(0.1 * data['train_data']['num_nodes']) # make it larger to fit the inductive nodes

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
    def test_tgb(self, 
                 full_data, 
                 test_mask, 
                 test_snapshots,
                 neg_sampler,
                 evaluator,  
                 embeddings,
                 metric,
                 split_mode):
        r"""
        TGB evaluation for discrete model
        1. load the queries from TGB
        2. remap the ts of the query into integers by discretizing the time
        prediction loop (per positive edge and its ns samples)
        a. unattach the node embed
        b. query the model 
        c. evaluate the prediction 
        embedding update step (for true edges)
        a. get all edges in the snapshot
        b. forward pass and update the embeddings

        Parameters:
            full_data: TGB dataset
            test_mask: mask for the test set
            test_snapshots: the test set in the form of snapshots
            neg_sampler: TGB negative sampler
            evaluator: TGB evaluator
            embeddings: the embeddings of the model
            metric: the metric to be used for evaluation
            split_mode: the split mode for the negative sampler
        """
        embeddings = embeddings.detach()
        perf_list = np.zeros(full_data['sources'][test_mask].shape[0])
        perf_idx = 0

        ts_list = test_snapshots['ts_map']
        ts_idx = 0
        update = True

        for pos_src, pos_dst, pos_t in zip(full_data['sources'][test_mask], full_data['destinations'][test_mask], full_data['timestamps'][test_mask]):

            #* if current timestamp exceeds the update the node embeddings
            if (pos_t > ts_list[ts_idx] and update == True):
                #* update the snapshot embedding
                pos_index = test_snapshots['edge_index'][ts_idx]
                pos_index = pos_index.long().to(args.device)
                z = self.model(pos_index, self.x)
                embeddings = self.model.update_hiddens_all_with(z)
                if (ts_idx >= len(ts_list) - 1):
                    update = False
                else:
                    ts_idx += 1

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

        perf_metrics = float(np.mean(perf_list))

        return perf_metrics, embeddings

    def run(self, seed=1):
        
        set_random(seed)
        optimizer = self.optimizer()  # @TODO: RiemannianAdam or Adam?!
        self.model.train()

        best_val = 0 
        best_test = 0
        best_epoch = 0


        #* set up TGB queries, this is only for val and test
        dataset = LinkPropPredDataset(name=args.dataset)
        full_data = dataset.full_data  
        metric = dataset.eval_metric
        neg_sampler = dataset.negative_sampler


        # get masks
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        evaluator = Evaluator(name=args.dataset)

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
                else:
                    #* feed the edge index from (t-1) into the model
                    prev_index = snapshot_list[snapshot_idx-1]
                    edge_index = prev_index.long().to(args.device)
                    z = self.model(edge_index, self.x)
                               

                if args.use_htc == 0:
                    epoch_loss = self.loss(z, pos_edge_index=edge_index,  neg_edge_index=neg_index)
                else:
                    epoch_loss = self.loss(z, pos_edge_index=edge_index, neg_edge_index=neg_index) + self.model.htc(z)
                
                epoch_loss.backward()
                optimizer.step()
                epoch_losses.append(epoch_loss.item())

                #* update the embedding for the final train snapshot
                if (snapshot_idx == (self.train_data['time_length']-1)):
                    pos_index = snapshot_list[snapshot_idx]
                    pos_index = pos_index.long().to(args.device)
                    z = self.model(pos_index, self.x)
                self.model.update_hiddens_all_with(z)
            
            average_epoch_loss = np.mean(epoch_losses)
            train_end_time = timeit.default_timer()

            #! validation and test code start up, use different logic than training
            
            # ==========================
            # Validation    
            # 1. load the queries from TGB
            # 2. remap the ts of the qeury
            self.model.eval()

            # id_map = self.train_data['id_map'] #remapped node id

            dataset.load_val_ns()

            #* steps for snapshots edges in test set
            #1. discretize the test set into snapshots
            #2. map from integer to unix ts
            val_start_time = timeit.default_timer()
            val_metrics, z = self.test_tgb(full_data, val_mask, self.val_data, neg_sampler, evaluator, z, metric, 'val')
            val_end_time = timeit.default_timer()

             # logging stats
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
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                test_metrics, z = self.test_tgb(full_data, test_mask, self.test_data, neg_sampler, evaluator, z, metric, 'test')
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