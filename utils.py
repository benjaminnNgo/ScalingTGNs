import numpy as np
import argparse
import sys
import torch

from torch_geometric.data import TemporalData


def list2csv(lst: list,
             fname: str,
             delimiter: str = ",",
             fmt: str = '%i'):
    out_list = np.array(lst)
    np.savetxt(fname, out_list, delimiter=delimiter,  fmt=fmt)



def get_args():
    parser = argparse.ArgumentParser('*** discretized training ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time', type=str, help='time granularity', default='daily')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv



def remove_duplicate_edges(data):

    src = data.src.numpy()
    dst = data.dst.numpy()
    ts = data.t.numpy()
    msg = data.msg.numpy()
    y = data.y.numpy()

    query = np.stack([src, dst, ts], axis=0)
    uniq, idx = np.unique(query, axis=1, return_index=True)
    print ("number of edges reduced from ", query.shape[1], " to ", uniq.shape[1])

    src = torch.from_numpy(src[idx])
    dst = torch.from_numpy(dst[idx])
    ts = torch.from_numpy(ts[idx])
    msg = torch.from_numpy(msg[idx])
    y = torch.from_numpy(y[idx])

    new_data = TemporalData(
            src=src,
            dst=dst,
            t=ts,
            msg=msg,
            y=y,
        )
    return new_data