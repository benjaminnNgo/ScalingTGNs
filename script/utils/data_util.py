import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle

from script.utils.TGS import TGS_Handler
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from script.utils.make_edges_new import get_edges, get_prediction_edges, get_prediction_edges_modified, get_new_prediction_edges, get_new_prediction_edges_modified


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    pedges, nedges = get_prediction_edges(undirected_edges)  # list
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data['edge_index_list'] = undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset):
    print('>> INFO: loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1

    gdata_list = []
    for edge_index in undirected_edges:
        gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
        gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

    data['gdata'] = gdata_list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_continuous_time_dataset(dataset, neg_sample):
    print("INFO: Loading a continuous-time dataset: {}".format(dataset))
    data = {}
    p_rawfile = '../data/input/continuous_time/{}_pedges_{}.pt'.format(dataset, neg_sample)  # positive edges
    n_rawfile = '../data/input/continuous_time/{}_nedges_{}.pt'.format(dataset, neg_sample)  # negative edges

    # positive edges
    pedge_index_list = torch.load(p_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    p_undirected_edges = get_edges(pedge_index_list)
    # negative edges
    nedge_index_list = torch.load(n_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    n_undirected_edges = get_edges(nedge_index_list)

    num_nodes = int(np.max(np.hstack(p_undirected_edges))) + 1  # only care about positive number of nodes

    pedges = get_prediction_edges_modified(p_undirected_edges)  # list
    nedges = get_prediction_edges_modified(n_undirected_edges)  # list

    new_pedges = get_new_prediction_edges_modified(p_undirected_edges, num_nodes)
    new_nedges = get_new_prediction_edges_modified(n_undirected_edges, num_nodes)

    data['edge_index_list'] = p_undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(pedge_index_list)
    data['weights'] = None
    print('>> INFO: Data: {}'.format(dataset))
    print('>> INFO: Total length: {}'.format(len(pedge_index_list)))
    print('>> INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def load_TGC_dataset(dataset):
    print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
    data = {}

    edgelist_rawfile = '../data/input/raw/edgelists/{}_edgelist.txt'.format(dataset)
    edgelist_df = pd.read_csv(edgelist_rawfile)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
    adj_time_list = []
    for ts in uniq_ts_list:
        # NOTE: this code does not use any node or edge features
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination']]
        ts_G = nx.from_pandas_edgelist(ts_edges, 'source', 'destination')
        ts_A = nx.to_scipy_sparse_array(ts_G)
        adj_time_list.append(ts_A)

    # Now, exactly like "load_vgrnn_dataset_det"
    print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
    edge_proc_start = time.time()
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('INFO: Processing finished! Elapsed time (sec.): {:.4}'.format(time.time() - edge_proc_start))
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('INFO: Data: {}'.format(dataset))
    print('INFO: Total length:{}'.format(len(edge_index_list)))
    print('INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10', neg_sample=''):


    # if cached, load directly
    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)  # the data will be saved here after generation.
    print("INFO: Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}.".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)
    
    # if not cached, to process and cached
    TGS_dataset_df = pd.read_csv("../data/TGS_available_datasets.csv")
    TGS_available_dataset = TGS_dataset_df['dataset'].tolist()
    
    print('INFO: data does not exits, processing ...')
    if dataset in ['enron10', 'dblp']:
        data = load_vgrnn_dataset(dataset)
    elif dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
        data = load_new_dataset(dataset)
    elif dataset in ['canVote', 'LegisEdgelist', 'wikipedia', 'UNtrade']:
        print("INFO: Loading a continuous-time dynamic graph dataset: {}".format(dataset))
        data = load_continuous_time_dataset(dataset, neg_sample)

    elif dataset in ['adex', 'aeternity', 'aion', "AMB"]:
        print("INFO: Loading a dynamic graph datasets for TG-Classification: {}".format(dataset))

        edgelist_rawfile = '../data/input/raw/edgelists/{}_edgelist.txt'.format(dataset)
        if os.path.exists(edgelist_rawfile):
            data = load_TGC_dataset(dataset)

    elif dataset in TGS_available_dataset:
        edgelist_rawfile = '../data/input/raw/edgelists/{}_edgelist.txt'.format(dataset)
        if os.path.exists(edgelist_rawfile):
            data = load_TGC_dataset(dataset)
        else:
            TGS_Handler("../data/input/tokens/raw/").creat_baseline_datasets(dataset)
            data = load_TGC_dataset(dataset)
    else:
        try:
            print("INFO: Loading a dynamic graph datasets from TGS")
            data = load_TGC_dataset(dataset)
        except Exception as e:
            raise ValueError("ERROR: Undefined dataset!")


    torch.save(data, filepath)
    print('INFO: Dataset is saved!')
    return data

def load_multiple_datasets(datasets_package_file,neg_sample):
    datasets_packages = []
    datasets_package_path = '../data/input/{}.txt'.format(datasets_package_file)
    print(datasets_package_path)
    if os.path.exists(datasets_package_path):
        print("File exists.")
    else:
        print("File does not exist.")

    try:
        with open(datasets_package_path, 'r') as file:
            for line in file:
                print("INFO: Dataset: {}".format(line))
                datasets_packages.append(loader(dataset=line.strip(), neg_sample=neg_sample))
    except Exception as e:
        print("ERROR: error in processing data pack {}".format(datasets_package_path))
        print(e)

    return datasets_packages



def process_data_gaps(directory,min_size = 10 ,max_size = 4000):
    columns = ["blockNumber", "timestamp", "tokenAddress", "from", "to", "value", "fileBlock"]
    dataset_feature_file = open('dataset_features.txt', 'w')
    dataset_feature_file.writelines(["filename, start, end, duration, max_gap,networkSize\n"])
    file_count = len(os.listdir(directory))
    counter = 0
    for filename in os.listdir(directory):
        counter += 1
        filepath = directory + "/" + filename
        file_size = os.path.getsize(filepath)/(1024*1024)
        try:
            if filename.endswith('.csv') and file_size>=min_size and file_size<=max_size:
                data = pd.read_csv(filepath, usecols=columns, index_col=False)

                timestamps = pd.to_datetime(data["timestamp"], unit="s").dt.date
                start = timestamps[0]
                end = timestamps.iloc[-1]
                time_difference = (end - start).days
                if time_difference < 20:
                    raise Exception("Token network last less than 20 days")

                unique_timestamps = timestamps.unique()
                tot_len = len(unique_timestamps)
                gaps = max(set([(unique_timestamps[i+1] - unique_timestamps[i]).days for i in range(tot_len-1)]))
                dataset_feature_file.writelines([filename, ",", str(start), ",", str(end), ",",str(time_difference),",", str(gaps),",",str(file_size),"\n"])
        except Exception as e:
            print("ERROR while processing {} due to\n {}".format(filename,e))

        print("Done processing {}/{}".format(counter,file_count))
    dataset_feature_file.close()

def select_datset_no_gap(filename,max_gap):
    dataset_df = pd.read_csv(filename)
    filtered_df = dataset_df[dataset_df[' max_gap'] <= max_gap]
    filtered_df.to_csv('dataset_no_gap_{}_day.csv'.format(max_gap), index=False)


def load_multiple_datasets(datasets_package_path=""):
    datasets_packages = []
    dataset_names = []

    print(datasets_package_path)
    if os.path.exists(datasets_package_path):
        print("Folder exists.")
    else:
        print("Folder does not exist.")
    i = 0
    text_path = "../data/input/data_list/{}".format(datasets_package_path)

    try:
        with open(text_path, 'r') as file:
            for line in file:
                print("INFO: Dataset: {}".format(line))
                datasets_packages.append(loader(dataset=line.strip()))
                dataset_names.append(line.strip())

    except Exception as e:
        print("ERROR: error in processing data pack {}".format(datasets_package_path))
        print(e)

    print("Number of dataset{}".format(len(datasets_packages)))
    return dataset_names, datasets_packages



def find_max_node_id (dataname):
    data_path = '../data/input/raw/edgelists/{}_edgelist.txt'.format(dataname)
    data_df = pd.read_csv(data_path)
    unique_node = set()
    unique_node.update(data_df['source'].tolist())
    unique_node.update(data_df['destination'].tolist())
    return max(unique_node)


def find_max_node_id_package(datasets_package_file):
    text_path = "../data/{}".format(datasets_package_file)
    max_id_dataset = []
    try:
        with open(text_path, 'r') as file:
            for dataset in file:
                max_id_dataset.append(find_max_node_id(dataset.strip()))
        return int(max(max_id_dataset))

    except Exception as e:
        print("ERROR: error in processing data pack {}".format(datasets_package_file))
        print(e)

if __name__ == '__main__':
    # process_data_gaps("E:/token/")
    # dataset_df = pd.read_csv("TGS_available_datasets.csv")
    # print(sum(dataset_df['networkSize'].tolist()))
    # print(max(dataset_df['networkSize'].tolist()))
    # select_datset_no_gap("dataset_features.txt",1)

    # print(find_max_node_id('unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0'))
    print(find_max_node_id_package("node_id_package.txt"))



