import timeit
import time 
import torch
import numpy as np
import argparse
import sys

from utils import remove_duplicate_edges
from negative_generator import NegativeEdgeGenerator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

"""
usage:

python tgbl-wiki_ns_gen.py -d tgbl-review -t minutely
"""


def get_args():
    parser = argparse.ArgumentParser('*** discretizing time steps from datasets ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time', type=str, help='time granularity', default='daily')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 



# 1. discretize dataset to only time granularities without time gap
# 2. keep both integer and unix timestamps
# 3. check for unique edges in test set 
# 4. generate the negative samples for test set and upload them
# 5. swap the ns sample from tgb with that of the new ones


"""
best way is to load processed ctdg data from tgb

1. remove the redundant edges in the test set 
"""



def main():
    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")

    # setting the required parameters
    num_neg_e_per_pos = 100 
    print ("generating {} negative samples per positive edge".format(num_neg_e_per_pos))
    #12000 #11000 #10000 #20 #100
    neg_sample_strategy = "hist_rnd" #"rnd"
    rnd_seed = 42

    args, _ = get_args()



    name = args.data
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    

    #! load DTDG timestamps for all edges
    time_scale = args.time #"hourly" #"minutely" #"daily" #"hourly"
    dataset_name = args.data
    ts_file = dataset_name + "_ts_" + time_scale + ".csv"
    dtdg_ts = np.genfromtxt(ts_file, delimiter=',', dtype=int)

    #* load the DTDG timestamps
    dtdg_ts = torch.from_numpy(dtdg_ts)
    if dtdg_ts.dtype != torch.int64:
        dtdg_ts = dtdg_ts.long()
    dataset.ts[val_mask] = dtdg_ts[val_mask]
    dataset.ts[test_mask] = dtdg_ts[test_mask]

    data = dataset.get_TemporalData()
    data_splits = {}
    data_splits['train'] = data[train_mask]
    data_splits['val'] = data[val_mask]
    data_splits['test'] = data[test_mask]

    data_splits['val'] = remove_duplicate_edges(data_splits['val'])
    data_splits['test'] = remove_duplicate_edges(data_splits['test'])

    # print ("there are edges in test set", data_splits['test'].src.shape)
    # quit()


    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # After successfully loading the dataset...
    if neg_sample_strategy == "hist_rnd":
        historical_data = data_splits["train"]
    else:
        historical_data = None

    neg_sampler = NegativeEdgeGenerator(
        dataset_name=name,
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        num_neg_e=num_neg_e_per_pos,
        strategy=neg_sample_strategy,
        rnd_seed=rnd_seed,
        historical_data=historical_data,
    )

    # generate evaluation set
    partial_path = "./"
    # generate validation negative edge set
    start_time = time.time()
    split_mode = "val"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path, suffix="ns_"+time_scale
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    # generate test negative edge set
    start_time = timeit.default_timer()
    split_mode = "test"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path, suffix="ns_"+time_scale
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {timeit.default_timer()- start_time: .4f}"

    )


if __name__ == "__main__":
    main()
