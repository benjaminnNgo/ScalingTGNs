import tgx
import timeit
import torch
import numpy as np
import argparse, sys
import sys
sys.path.append('../')


from data_script.dtdg_neg_generator import NegativeEdgeGenerator
from torch_geometric.data import TemporalData
from typing import Optional, Dict, Any, Tuple
from utils.utils_func import generate_splits, convert2Torch

def get_args():
    parser = argparse.ArgumentParser('*** negative sample generating options ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time_scale', type=str, default='month', help='Time scale to discretize a TGB dataset.',
                    choices=['minutely', 'hourly', 'daily', 'weekly', 'monthly', 'yearly', 'biyearly'])
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv




def main():
    args, _ = get_args()

    r"""
    Generate negative edges for the validation or test phase
    """
    print("*** Negative Sample Generation ***")

    
    neg_sample_strategy = "hist_rnd" #"rnd"
    rnd_seed = 42

    name = args.data #"CanParl" #"uci"

    args.data = args.data.lower()
    # load the DTDG dataset here
    if (args.data == "uci"):
        dataset = tgx.builtin.uci() 
    elif (args.data == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (args.data == "enron"):
        dataset = tgx.builtin.enron()
    elif (args.data == "contacts"):
        dataset = tgx.builtin.contacts()
    elif (args.data == "mooc"):
        dataset = tgx.builtin.mooc()
    elif (args.data == "social_evo"):
        dataset = tgx.builtin.social_evo()
    elif (args.data == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (args.data == "lastfm"):
        dataset = tgx.builtin.lastfm()
    else:
        raise ValueError("Invalid dataset name")
    

    ctdg = tgx.Graph(dataset)
    time_scale = args.time_scale #"weekly"
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    
    # setting the required parameters
    #! setting the number of negative edges per positive edge
    num_neg_e_per_pos = min(dtdg.total_nodes(), 1000) #1000

    print ("discretize to ", time_scale)
    print ("there is time gap, ", dtdg.check_time_gap())
    print ("generating negative samples for ", name, " with strategy ", neg_sample_strategy, " and num_neg_e_per_pos ", num_neg_e_per_pos)


    full_data = dtdg.export_full_data()
    sources = full_data["sources"]
    destinations = full_data["destinations"]
    timestamps = full_data["timestamps"]

    val_ratio = 0.15
    test_ratio = 0.15
    train_mask, val_mask, test_mask = generate_splits(full_data,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_mask = torch.from_numpy(train_mask)
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)
    src, dst, ts = convert2Torch(sources, destinations, timestamps)


    data = TemporalData(
        src=src,
        dst=dst,
        t=ts,
    )

    data_splits = {}
    data_splits['train'] = data[train_mask]
    data_splits['val'] = data[val_mask]
    data_splits['test'] = data[test_mask]

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
    start_time = timeit.default_timer()
    split_mode = "val"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {timeit.default_timer() - start_time: .4f}"
    )

    # generate test negative edge set
    start_time = timeit.default_timer()
    split_mode = "test"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {neg_sample_strategy}"
    )
    neg_sampler.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {timeit.default_timer()- start_time: .4f}"

    )


if __name__ == "__main__":
    main()
