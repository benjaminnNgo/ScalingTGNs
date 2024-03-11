
import timeit
import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
import wandb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils_func import generate_splits, convert2Torch, set_random, get_snapshot_batches
from utils.data_util import load_dtdg
from torch_geometric.data import TemporalData
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler


from models.tgn.decoder import LinkPredictor
from models.tgn.early_stopping import  EarlyStopMonitor
from models.tgn.emb_module import GraphAttentionEmbedding
from models.tgn.msg_func import IdentityMessage
from models.tgn.msg_agg import LastAggregator
from models.tgn.neighbor_loader import LastNeighborLoader
from models.tgn.memory_module import TGNMemory


def get_args():
    parser = argparse.ArgumentParser('*** TGN on discrete datasets ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time_scale', type=str, default='month', help='Time scale to discretize a TGB dataset.',
                    choices=['minutely', 'hourly', 'daily', 'weekly', 'monthly', 'yearly','biyearly'])
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--max_epoch', type=int, help='Number of epochs', default=300)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=20)
    parser.add_argument('--num_runs', type=int, help='Number of iteration runs', default=1)
    parser.add_argument("--wandb", action="store_true", default=False, help="now using wandb")
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def train(index_dict, tids, data):
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        index_dict: a dictionary that contains the start and end index of each snapshot
        tids: the timestamps to train
        data: a dataset object
    Returns:
        total_loss: the total loss of the training
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for tid in tids:
        idx_s = index_dict[tid][0]
        idx_e = index_dict[tid][1]

        src = data.src[idx_s:idx_e].to(device)
        pos_dst = data.dst[idx_s:idx_e].to(device)
        t = data.t[idx_s:idx_e].to(device)
        msg = data.msg[idx_s:idx_e].to(device)

        optimizer.zero_grad()

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * src.size(0)

    return total_loss / train_data.num_events


@torch.no_grad()
def test(index_dict, tids, data, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        index_dict: a dictionary that contains the start and end index of each snapshot
        tids: the timestamps to evaluate
        data: a dataset object
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []


    for tid in tids:
        idx_s = index_dict[tid][0]
        idx_e = index_dict[tid][1]

        pos_src = data.src[idx_s:idx_e]
        pos_dst = data.dst[idx_s:idx_e]
        pos_t = data.t[idx_s:idx_e]
        pos_msg = data.msg[idx_s:idx_e]

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

TIME_SCALE = args.time_scale
use_wandb = args.wandb
DATA = args.data 
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.max_epoch
SEED = args.seed
set_random(args.seed)
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience #100 #20 # args.patience
NUM_RUNS = args.num_runs
NUM_NEIGHBORS = 10


if args.wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="utg",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "batch_size": args.bs,
        "architecture":"TGN",
        "dataset": args.data,
        "time granularity": args.time_scale,
        "memory_dim": args.mem_dim,
        "time_dim": args.time_dim,
        "embedding_dim": args.emb_dim,
        }
    )



MODEL_NAME = 'TGN'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtdg, ts_list = load_dtdg(DATA, TIME_SCALE)

full_data = dtdg.export_full_data()
sources = full_data["sources"]
destinations = full_data["destinations"]
timestamps = full_data["timestamps"]

#! get a list of snapshot batches from the timestamps
index_dict = get_snapshot_batches(timestamps)


val_ratio = 0.15
test_ratio = 0.15
train_mask, val_mask, test_mask = generate_splits(full_data,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
)

#TODO currently assumes there is no edge feature for the discrete dataset 

train_mask = torch.from_numpy(train_mask)
val_mask = torch.from_numpy(val_mask)
test_mask = torch.from_numpy(test_mask)
feat_size = 1 #args.emb_dim
msg = np.zeros((sources.shape[0], feat_size))
msg = torch.from_numpy(msg).float()

src, dst, ts = convert2Torch(sources, destinations, timestamps)
data = TemporalData(
    src=src,
    dst=dst,
    t=ts,
    msg=msg,
)
data = data.to(device)
metric = "mrr"


#data loaders
train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_start = train_data.t[0].item()
train_end = train_data.t[-1].item()
val_start = val_data.t[0].item()
val_end = val_data.t[-1].item()
test_start = test_data.t[0].item()
test_end = test_data.t[-1].item()

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name="tgbl-wiki")
neg_sampler = NegativeEdgeSampler(dataset_name=DATA, strategy="hist_rnd")


for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    set_random(run_idx + SEED)

    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)



    #* load the val negative samples
    neg_sampler.load_eval_set(fname=DATA + "_val_ns.pkl", split_mode="val",)

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):

        # training
        start_epoch_train = timeit.default_timer()
        tids = range(train_start, train_end + 1)  # all inclusive for train_start and train_end
        loss = train(index_dict, tids, data)
        train_time = timeit.default_timer() - start_epoch_train
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {train_time: .4f}"
        )

        # validation
        start_val = timeit.default_timer()
        tids = range(val_start, val_end + 1)  # all inclusive for val_start and val_end
        perf_metric_val = test(index_dict, tids, data, neg_sampler, split_mode="val")
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        val_time = timeit.default_timer() - start_val
        print(f"\tValidation: Elapsed time (s): {val_time: .4f}")
        val_perf_list.append(perf_metric_val)

        if (args.wandb):
            wandb.log({"train_loss": loss,
                    "val_" + metric: perf_metric_val,
                    "train time": train_time,
                    "val time": val_time,
                    })
        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)
    best_epoch = early_stopper.best_epoch
    best_val = early_stopper.best_sofar

    #* load the val negative samples
    neg_sampler.load_eval_set(fname=DATA + "_test_ns.pkl", split_mode="test",)

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(index_dict, test_start, test_end, data, neg_sampler, split_mode="test")

    print(f"INFO: Test")
    print(f"\tBest Epoch: {best_epoch}")
    print(f"\tBest Val: {metric}: {best_val: .4f}")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
 

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")