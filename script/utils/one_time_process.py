"""
Functions that might be used once for processing purposes
"""

import pandas as pd
import numpy as np


def pre_proc_RedditB(filename, output_filename):
    """
    preprocess RedditB dataset
    this include conversion of node-id (which are currently string) to integer ID
    """
    edgelist_df = pd.read_csv(filename)
    print(edgelist_df.head())
    unique_nodes = list(set(edgelist_df['source'].tolist() + edgelist_df['destination'].tolist()))

    node_id_dict = {}
    for idx, node in enumerate(unique_nodes):
        node_id_dict[node] = idx

    source_new, destination_new = [], []
    for idx, row in edgelist_df.iterrows():
        source_new.append(node_id_dict[row['source']])
        destination_new.append(node_id_dict[row['destination']])

    edgelist_df = edgelist_df.drop(['source', 'destination'], axis=1)
    edgelist_df['source'] = source_new
    edgelist_df['destination'] = destination_new

    print(edgelist_df.head())

    print("INFO: RedditB is pre-processed. File is saved at {}".format(output_filename))
    edgelist_df.to_csv(output_filename, index=False)
    

def main():
    """
    Main
    """
    # NOTE: inside script/utils/ --> python one_time_process.py
    filename = f'../../data/input/raw/RedditB/RedditB_edgelist_original.txt'
    output_filename = f'../../data/input/raw/RedditB/RedditB_edgelist.txt'
    print("INFO: Input file is at: {}".format(filename))
    pre_proc_RedditB(filename, output_filename)


if __name__ == '__main__':
    main()
