import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
import pickle
import shutil
import datetime as dt


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


# def load_vgrnn_dataset(dataset):
#     assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
#     print('>> loading on vgrnn dataset')
#     with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
#         adj_time_list = pickle.load(handle, encoding='iso-8859-1')
#     print('>> generating edges, negative edges and new edges, wait for a while ...')
#     data = {}
#     edges, biedges = mask_edges_det(adj_time_list)  # list
#     pedges, nedges = mask_edges_prd(adj_time_list)  # list
#     new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
#     print('>> processing finished!')
#     assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
#     edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
#     for t in range(len(biedges)):
#         edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
#         pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
#         nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
#         new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
#         new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

#     data['edge_index_list'] = edge_index_list
#     data['pedges'], data['nedges'] = pedges_list, nedges_list
#     data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
#     data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

#     data['time_length'] = len(edge_index_list)
#     data['weights'] = None
#     print('>> data: {}'.format(dataset))
#     print('>> total length:{}'.format(len(edge_index_list)))
#     print('>> number nodes: {}'.format(data['num_nodes']))
#     return data


# def load_new_dataset(dataset):
#     print('>> loading on new dataset')
#     data = {}
#     rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
#     edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
#     undirected_edges = get_edges(edge_index_list)
#     num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
#     pedges, nedges = get_prediction_edges(undirected_edges)  # list
#     new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

#     data['edge_index_list'] = undirected_edges
#     data['pedges'], data['nedges'] = pedges, nedges
#     data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
#     data['num_nodes'] = num_nodes
#     data['time_length'] = len(edge_index_list)
#     data['weights'] = None
#     print('>> INFO: data: {}'.format(dataset))
#     print('>> INFO: total length: {}'.format(len(edge_index_list)))
#     print('>> INFO: number nodes: {}'.format(data['num_nodes']))
#     return data


# def load_vgrnn_dataset_det(dataset):
#     assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
#     print('>> loading on vgrnn dataset')
#     with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
#         adj_time_list = pickle.load(handle, encoding='iso-8859-1')
#     print('>> generating edges, negative edges and new edges, wait for a while ...')
#     data = {}
#     edges, biedges = mask_edges_det(adj_time_list)  # list
#     pedges, nedges = mask_edges_prd(adj_time_list)  # list
#     new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
#     print('>> processing finished!')
#     assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
#     edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
#     for t in range(len(biedges)):
#         edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
#         pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
#         nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
#         new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
#         new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

#     data['edge_index_list'] = edge_index_list
#     data['pedges'], data['nedges'] = pedges_list, nedges_list
#     data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
#     data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

#     data['time_length'] = len(edge_index_list)
#     data['weights'] = None
#     print('>> data: {}'.format(dataset))
#     print('>> total length:{}'.format(len(edge_index_list)))
#     print('>> number nodes: {}'.format(data['num_nodes']))
#     return data


# def load_new_dataset_det(dataset):
#     print('>> INFO: loading on new dataset')
#     data = {}
#     rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
#     edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
#     undirected_edges = get_edges(edge_index_list)
#     num_nodes = int(np.max(np.hstack(undirected_edges))) + 1

#     gdata_list = []
#     for edge_index in undirected_edges:
#         gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
#         gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

#     data['gdata'] = gdata_list
#     data['num_nodes'] = num_nodes
#     data['time_length'] = len(edge_index_list)
#     data['weights'] = None
#     print('>> INFO: data: {}'.format(dataset))
#     print('>> INFO: total length: {}'.format(len(edge_index_list)))
#     print('>> INFO: number nodes: {}'.format(data['num_nodes']))
#     return data


# def load_continuous_time_dataset(dataset, neg_sample):
#     print("INFO: Loading a continuous-time dataset: {}".format(dataset))
#     data = {}
#     p_rawfile = '../data/input/continuous_time/{}_pedges_{}.pt'.format(dataset, neg_sample)  # positive edges
#     n_rawfile = '../data/input/continuous_time/{}_nedges_{}.pt'.format(dataset, neg_sample)  # negative edges

#     # positive edges
#     pedge_index_list = torch.load(p_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
#     p_undirected_edges = get_edges(pedge_index_list)
#     # negative edges
#     nedge_index_list = torch.load(n_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
#     n_undirected_edges = get_edges(nedge_index_list)

#     num_nodes = int(np.max(np.hstack(p_undirected_edges))) + 1  # only care about positive number of nodes

#     pedges = get_prediction_edges_modified(p_undirected_edges)  # list
#     nedges = get_prediction_edges_modified(n_undirected_edges)  # list

#     new_pedges = get_new_prediction_edges_modified(p_undirected_edges, num_nodes)
#     new_nedges = get_new_prediction_edges_modified(n_undirected_edges, num_nodes)

#     data['edge_index_list'] = p_undirected_edges
#     data['pedges'], data['nedges'] = pedges, nedges
#     data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
#     data['num_nodes'] = num_nodes
#     data['time_length'] = len(pedge_index_list)
#     data['weights'] = None
#     print('>> INFO: Data: {}'.format(dataset))
#     print('>> INFO: Total length: {}'.format(len(pedge_index_list)))
#     print('>> INFO: Number nodes: {}'.format(data['num_nodes']))
#     return data


# def load_TGC_dataset(dataset):
#     print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
#     data = {}
#     edgelist_rawfile = '../data/input/raw/{}/{}_edgelist.txt'.format(dataset, dataset)
#     edgelist_df = pd.read_csv(edgelist_rawfile)
#     uniq_ts_list = np.unique(edgelist_df['snapshot'])
#     print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
#     adj_time_list = []
#     for ts in uniq_ts_list:
#         # NOTE: this code does not use any node or edge features
#         ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination']]
#         ts_G = nx.from_pandas_edgelist(ts_edges, 'source', 'destination')
#         ts_A = nx.to_scipy_sparse_array(ts_G)
#         adj_time_list.append(ts_A)

#     # Now, exactly like "load_vgrnn_dataset_det"
#     print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
#     edge_proc_start = time.time()
#     data = {}
#     edges, biedges = mask_edges_det(adj_time_list)  # list
#     pedges, nedges = mask_edges_prd(adj_time_list)  # list
#     new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
#     print('INFO: Processing finished! Elapsed time (sec.): {:.4}'.format(time.time() - edge_proc_start))
#     assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
#     edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
#     for t in range(len(biedges)):
#         edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
#         pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
#         nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
#         new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
#         new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

#     data['edge_index_list'] = edge_index_list
#     data['pedges'], data['nedges'] = pedges_list, nedges_list
#     data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
#     data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

#     data['time_length'] = len(edge_index_list)
#     data['weights'] = None
#     print('INFO: Data: {}'.format(dataset))
#     print('INFO: Total length:{}'.format(len(edge_index_list)))
#     print('INFO: Number nodes: {}'.format(data['num_nodes']))
#     return data


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
    print('INFO: data does not exits, processing ...')
    # if dataset in ['enron10', 'dblp']:
    #     data = load_vgrnn_dataset(dataset)
    # elif dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
    #     data = load_new_dataset(dataset)
    # elif dataset in ['canVote', 'LegisEdgelist', 'wikipedia', 'UNtrade']:
    #     print("INFO: Loading a continuous-time dynamic graph dataset: {}".format(dataset))
    #     data = load_continuous_time_dataset(dataset, neg_sample)
    # elif dataset in ['adex', 'aeternity', 'aion', 'aragon', 'bancor', 'centra', 'cindicator', 
    #                  'coindash', 'dgd', 'iconomi',  'mathoverflow', 'RedditB', 'CollegeMsg']:
    #     print("INFO: Loading a dynamic graph datasets for TG-Classification: {}".format(dataset))
    #     data = load_TGC_dataset(dataset)
    # else:
    #     raise ValueError("ERROR: Undefined dataset!")
    # torch.save(data, filepath)
    # print('INFO: Dataset is saved!')
    # return data

root_path = "../../data/all_network/"
# timeseries_file_path = "../data/all_network/TimeSeries/"
timeseries_file_path = "/network/scratch/r/razieh.shirzadkhani/fm_data/selected/"

def creatBaselineDatasets(file, normalization=False):
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0
    selectedNetwork = pd.read_csv((timeseries_file_path + file), usecols = ["timestamp", "from", "to", "value"])
    selectedNetwork = selectedNetwork.rename(columns={"timestamp" : "date"})
    # selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=',', header = 0, usecols = ["from", "to", "timestamp", "value"], names=["from", "to", "date", "value"])
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()

    print(f"{file} -- {window_start_date} -- {data_last_date}")

    print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
    # check if the network has more than 20 days of data
    if ((data_last_date - window_start_date).days < minValidDuration):
        print(file + "Is not a valid network")
        shutil.move(root_path + file, root_path + "Invalid/" + file)
        return

    # normalize the edge weights for the graph network {0-9}
    max_transfer = float(selectedNetwork['value'].max())
    min_transfer = float(selectedNetwork['value'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # value normalization
    if normalization == False:
        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    # Graph Generation Process and Labeling

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nRemaining Process  {} ".format(

            (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
        indx += 1

        # select window data
        window_end_date = window_start_date + dt.timedelta(days=windowSize)
        selectedNetworkInGraphDataWindow = selectedNetwork[
            (selectedNetwork['date'] >= window_start_date) & (
                    selectedNetwork['date'] < window_end_date)]

        # select labeling data
        label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
            days=gap) + dt.timedelta(
            days=lableWindowSize)
        label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
        selectedNetworkInLbelingWindow = selectedNetwork[
            (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

        # generating the label for this window
        # 1 -> Increading Transactions 0 -> Decreasing Transactions
        label = 1 if (len(selectedNetworkInLbelingWindow) - len(
            selectedNetworkInGraphDataWindow)) > 0 else 0

        # Storing the new snapshot data after processing
        selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(Snapshot=indx)
        csv_file_path = "../data/all_network/TimeSeries/Baseline/" + file
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
        else:
            existing_df = pd.DataFrame()

        appended_df = existing_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
        appended_df.to_csv(csv_file_path, index=False)
        # ------------------------------------------------
        # Storing each snapshot label data
        label_csv_file_path = "../data/all_network/TimeSeries/Baseline/labels" + file
        if os.path.exists(label_csv_file_path):
            existing_label_df = pd.read_csv(label_csv_file_path)
        else:
            existing_label_df = pd.DataFrame()

        appended_label_df = existing_label_df.append(label, ignore_index=True)
        appended_label_df.to_csv(label_csv_file_path, index=False)
        # --------------------------------------------------

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed!")


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

    print("BAO:Number of dataset{}".format(len(datasets_packages)))
    return datasets_packages


def process_data_gaps(directory):

    columns = ["blockNumber", "timestamp", "tokenAddress", "from", "to", "value", "fileBlock"]
    file1 = open('dataset_features.txt', 'w')
    file1.writelines(["filename, start, end, duration, max_gap"])

    for filename in os.listdir(directory):
        filepath = directory + "/" + filename  

        if filename.endswith('.csv'):
            data = pd.read_csv(filepath, usecols=columns, index_col=False)
            timestamps = pd.to_datetime(data["timestamp"], unit="s").dt.date
            start = timestamps[0]
            end = timestamps.iloc[-1]
            time_difference = (end - start).days

            unique_timestamps = timestamps.unique()
            tot_len = len(unique_timestamps)
            gaps = max(set([(unique_timestamps[i+1] - unique_timestamps[i]).days for i in range(tot_len-1)]))
            file1.writelines([filename, ",", str(start), ",", str(end), ",",str(time_difference),",", str(gaps) ,"\n"])            
    file1.close()


if __name__ == '__main__':
    creatBaselineDatasets("unnamed_token_15_0x0000000000095413afc295d19edeb1ad7b71c952.csv")
#     process_data_gaps("/network/scratch/r/razieh.shirzadkhani/fm_data")