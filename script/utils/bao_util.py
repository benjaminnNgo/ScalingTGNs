import json
import pandas as pd
import statistics as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx



def generate_table_training(df_single,df_foundation,datasets_list,df_TGS,df_baseline):
    dataset_single = []
    dataset_foundation = []
    seeds = set([710,720,800])
    missing_dict = {}
    missing_foundation =[]
    count = 0
    improvement = 0
    found_table = {}
    single_table = {}
    diffs= []
    for data in datasets_list:
        row_token = df_TGS[df_TGS['dataset'] == data]
        rows_single = df_single[df_single['dataset'] == data]
        rows_found = df_foundation[df_foundation['dataset'] == data]
        rows_baseline = df_baseline[df_baseline['dataset'] == data]

        token_name = row_token['token_name'].values[0]
        auc_found = rows_found['auc'].values[0]
        std_found = rows_found['std'].values[0]
        persistence = round(rows_baseline['auc'].values[0],3)

        auc_single = round(stats.mean(rows_single['test_auc'].tolist()),3)
        std_single = round(stats.stdev(rows_single['test_auc'].tolist()),3)
        improvement = auc_found - auc_single
        diffs.append(improvement)
        row ="{} & {} & {} $\pm$ \scriptsize{} & {} $\pm$ \scriptsize{} \\\\".format(token_name,persistence,auc_single,std_single,auc_found,std_found)

        if auc_found >= auc_single:
            dataset_foundation.append(data)
            diff = auc_found - auc_single

            if diff in found_table:
                diff+= 0.0000001
            found_table[diff] =row
        else:
            dataset_single.append(data)
            diff = auc_single - auc_found

            if diff in single_table:
                diff+= 0.0000001
            single_table[diff] =row

    dict = {"single": dataset_single, "foundation": dataset_foundation}
    with open('performace_split.json', 'w') as json_file:
        json.dump(dict, json_file, indent=4)
    found_keys = list(found_table.keys())
    print(len(found_keys))
    found_keys = sorted(found_keys, reverse=True)
    for key in found_keys:
        print(found_table[key])

    print("=="*30)

    single_key = list(single_table.keys())
    print(len(single_key))
    single_key = sorted(single_key, reverse=True)
    for key in single_key:
        print(single_table[key])

    print(stats.mean(diffs))
    print(stats.stdev(diffs))


def draw_distribution():
    with open('performace_split.json', 'r') as json_file:
        data = json.load(json_file)

    single_dataset = data['single']
    foundation_dataset = data['foundation']

    single = []
    foundation = []
    column = 'surprise'
    name = 'surprise'

    for data in single_dataset:
        data_row = df_TGS_stats[df_TGS_stats['dataset'] == data]
        single.append(data_row[column].values[0])

    for data in foundation_dataset:
        data_row = df_TGS_stats[df_TGS_stats['dataset'] == data]
        foundation.append(data_row[column].values[0])

    plt.hist(single, bins=30, alpha=0.5, label='Single perform better')
    plt.hist(foundation, bins=30, alpha=0.5, label='Foundation perform better')

    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'{name} distribution of 2 packages of datasets')
    plt.legend()
    plt.show()

def new_node_per_snapshot(dataset):
    path = "E:/TGS_edgelists/edgelists/"
    dataset_df = pd.read_csv("{}{}_edgelist.txt".format(path,dataset))
    exit_node = set()
    snapshot_idxs = dataset_df['snapshot'].unique()
    new_node_count = []
    for i in snapshot_idxs:
        edge_list = dataset_df[dataset_df['snapshot'] == i]
        snapshot_nodes = set(edge_list['source'].unique().tolist() + edge_list['destination'].unique().tolist())
        new_node_count.append(len(snapshot_nodes.difference(exit_node)))
        exit_node.update(snapshot_nodes)

    # print(new_node_count)
    return stats.mean(new_node_count)

def triangle_count(dataset):
    path = "E:/TGS_edgelists/edgelists/"
    dataset_df = pd.read_csv("{}{}_edgelist.txt".format(path, dataset))
    snapshot_idxs = dataset_df['snapshot'].unique()
    diameter_list = []
    for i in snapshot_idxs:
        ts_edges = dataset_df.loc[dataset_df['snapshot'] == i, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination')
        diameter_list.append(sum(nx.triangles(ts_G).values()) // 3)
    return stats.mean(diameter_list)

def average_clustering(dataset):
    path = "E:/TGS_edgelists/edgelists/"
    dataset_df = pd.read_csv("{}{}_edgelist.txt".format(path, dataset))
    snapshot_idxs = dataset_df['snapshot'].unique()
    average_cluster = []
    for i in snapshot_idxs:
        ts_edges = dataset_df.loc[dataset_df['snapshot'] == i, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges,  source='source', target='destination')
        average_cluster.append(nx.average_clustering(ts_G))
    return stats.mean(average_cluster)

#need to compute
def average_transitivity(dataset):
    path = "E:/TGS_edgelists/edgelists/"
    dataset_df = pd.read_csv("{}{}_edgelist.txt".format(path, dataset))
    snapshot_idxs = dataset_df['snapshot'].unique()
    average_transitivity = []
    for i in snapshot_idxs:
        ts_edges = dataset_df.loc[dataset_df['snapshot'] == i, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination')
        average_transitivity.append(nx.transitivity(ts_G))
    return stats.mean(average_transitivity)







if __name__ == "__main__":
    df_single = pd.read_csv("../../data/output/single_model_train_set/results.csv")
    df_foundation = pd.read_csv("../../data/output/fm-64-results.csv")
    datasets_list = pd.read_csv("../../data/data_package/datasets_package_64.txt").iloc[:, 0].values
    df_TGS = pd.read_csv("../../data/TGS_available_datasets.csv")
    df_baseline = pd.read_csv("../../data/output/baselinemodel.csv")
    df_TGS_stats = pd.read_csv("../../data/TGS_stats.csv")

    # generate_table_training(df_single, df_foundation, datasets_list, df_TGS, df_baseline)
    new_metric = []
    for dataset in df_TGS_stats['dataset'].tolist():
        print(dataset)
        token = df_TGS[df_TGS['dataset'] ==dataset]['token_name'].values[0]
        new_metric.append(count_arborescences(token))

    df_TGS_stats['arborescences'] = new_metric
    df_TGS_stats.to_csv("../../data/TGS_stats.csv",index=False)