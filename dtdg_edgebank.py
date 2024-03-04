import timeit
import numpy as np
from tqdm import tqdm
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from tgb.linkproppred.evaluate import Evaluator
from models.edgebank_predictor import EdgeBankPredictor
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler


# internal imports
from utils.utils_func import generate_splits, set_random, get_snapshot_batches
from utils.data_util import load_dtdg



def get_args():
    parser = argparse.ArgumentParser('*** Edgebank on discrete datasets ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time_scale', type=str, default='month', help='Time scale to discretize a TGB dataset.',
                    choices=['minutely', 'hourly', 'daily', 'weekly', 'monthly', 'yearly'])
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='unlimited', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv



def test(index_dict, tids, data, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges
    star and end idx is both inclusive
    Parameters:
        data: a dataset object
        tids: the timestamps to evaluate
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    perf_list = []
    for tid in tids:
        idx_s = index_dict[tid][0]
        idx_e = index_dict[tid][1]

        pos_src = data['sources'][idx_s:idx_e]
        pos_dst = data['destinations'][idx_s:idx_e]
        pos_t = data['timestamps'][idx_s:idx_e]

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
            
        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = float(np.mean(perf_list))

    return perf_metrics


start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args()
print("INFO: Arguments:", args)


SEED = args.seed  # set the random seed for consistency
DATA = args.data 
SEED = args.seed
set_random(args.seed)
TIME_SCALE = args.time_scale
MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = args.bs
K_VALUE = args.k_value
TIME_WINDOW_RATIO = args.time_window_ratio
MODEL_NAME = 'EdgeBank'


dtdg, ts_list = load_dtdg(DATA, TIME_SCALE)
full_data = dtdg.export_full_data()
sources = full_data["sources"]
destinations = full_data["destinations"]
timestamps = full_data["timestamps"]
metric = "mrr"

index_dict = get_snapshot_batches(timestamps)
val_ratio = 0.15
test_ratio = 0.15
train_mask, val_mask, test_mask = generate_splits(full_data,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
)

train_start = timestamps[train_mask][0]
train_end = timestamps[train_mask][-1]
val_start = timestamps[val_mask][0]
val_end = timestamps[val_mask][-1]
test_start = timestamps[test_mask][0]
test_end = timestamps[test_mask][-1]



data = {
    "sources": sources,
    "destinations": destinations,
    "timestamps": timestamps,
}


#data for memory in edgebank
hist_src = np.concatenate([data['sources'][train_mask]])
hist_dst = np.concatenate([data['destinations'][train_mask]])
hist_ts = np.concatenate([data['timestamps'][train_mask]])

# Set EdgeBank with memory updater
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO)

print("==========================================================")
print(f"============*** {MODEL_NAME}: {MEMORY_MODE}: {DATA} ***==============")
print("==========================================================")

evaluator = Evaluator(name="tgbl-wiki") #use same evaluator as tgbl_wiki
neg_sampler = NegativeEdgeSampler(dataset_name=DATA, strategy="hist_rnd")


# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{MEMORY_MODE}_{DATA}_results.json'

# ==================================================== Test
# loading the validation negative samples
neg_sampler.load_eval_set(fname=DATA + "_val_ns.pkl", split_mode="val",)
# testing ...
start_val = timeit.default_timer()
tids = range(val_start, val_end + 1) #all inclusive for val_start and val_end
perf_metric_test = test(index_dict, tids, data, neg_sampler, split_mode='val')
end_val = timeit.default_timer()

print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tval: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_val
print(f"\tval: Elapsed Time (s): {test_time: .4f}")


# ==================================================== Test
# loading the test negative samples
neg_sampler.load_eval_set(fname=DATA + "_test_ns.pkl", split_mode="test",)

# testing ...
start_test = timeit.default_timer()
tids = range(test_start, test_end + 1) #all inclusive for test_start and test_end
perf_metric_test = test(index_dict, tids, data, neg_sampler, split_mode='test')
end_test = timeit.default_timer()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tTest: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_test
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
