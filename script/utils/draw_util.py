import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stat
import matplotlib.pyplot as plt
import plotly.express as px


def draw_avg_rank():
    rank_htgn = pd.read_csv("../utils/htgn_results.csv")
    rank_gclstm = pd.read_csv("../utils/gclstm_results.csv")

    htgn = rank_htgn[rank_htgn['method'] == 'avg_rank']
    gclstm = rank_gclstm[rank_gclstm['method'] == 'avg_rank']

    foundation_names = ["1", "2", "4", "8", "16", "32", "64"]
    htgn_list = []
    gclstm_list = []
    for foundation_name in foundation_names:
        htgn_list.append(float(htgn[foundation_name]))
        gclstm_list.append(float(gclstm[foundation_name]))

    data = pd.DataFrame({
        "method": foundation_names,
        "htgn": htgn_list,
        "gclstm": gclstm_list
    })
    plt.figure(figsize=(10, 6))

    # Plot the first line
    sns.lineplot(x='method', y='htgn', data=data, label='HTGN', linewidth=3)

    # Plot the second line
    sns.lineplot(x='method', y='gclstm', data=data, label='GCLSTM', linewidth=3)

    # Add a title and labels
    plt.xlabel('Number of networds', fontsize=20)
    plt.ylabel('Average rank', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show the legend

    plt.legend(fontsize='x-large')

    plt.savefig("../../pic/average_rank.pdf")

    # Display the plot
    plt.show()


def draw_dataset_distribution(columns="age"):
    if columns == "age":
        scale = False
        xaxis = "Days"
        filename = "day"
    elif columns == "novelty":
        scale = False
        xaxis = "Novelty score"
        filename = "novelty"
    elif columns == "node_count":
        scale = True
        xaxis = "Nodes"
        filename = "node"
    else:
        scale = True
        xaxis = "Transactions"
        filename = "transaction"

    TGS_stats_df = pd.read_csv('../TGS_stats.csv')
    list = TGS_stats_df[columns].tolist()

    plt.figure(figsize=(10, 9))  # Width, height in inches

    g = sns.histplot(list, log_scale=scale, bins=20)
    g.set_xlabel(xaxis, fontsize=30)
    g.set_ylabel('Frequency', fontsize=30)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=30)

    plt.savefig("../../pic/{}_TGS.pdf".format(filename))

    plt.show()


plt.show()


def file_to_array(file_path):
    with open(file_path, 'r') as file:
        # Read all lines from the file and strip the newline characters
        array = [line.strip() for line in file.readlines()]
    return array


def draw_train_distribution(columns="node_count"):
    if columns == "age":
        scale = False
        xaxis = "Days"
        filename = "day"
    elif columns == "novelty":
        scale = False
        xaxis = "Novelty score"
        filename = "novelty"
    elif columns == "node_count":
        scale = True
        xaxis = "Nodes"
        filename = "node"
    else:
        scale = True
        xaxis = "Transactions"
        filename = "transaction"

    TGS_stats_df = pd.read_csv('../TGS_stats.csv')
    train_data = file_to_array("../../data/data_package/datasets_package_64.txt")

    train_stats = TGS_stats_df[TGS_stats_df['dataset'].isin(train_data)][columns].tolist()

    # print
    plt.figure(figsize=(10, 9))  # Width, height in inches

    g = sns.histplot(train_stats, log_scale=scale, bins=20)
    g.set_xlabel(xaxis, fontsize=30)
    g.set_ylabel('Frequency', fontsize=30)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=30)

    plt.savefig("../../pic/train_{}_TGS.pdf".format(filename))

    plt.show()


def draw_test_distribution(columns="novelty"):
    if columns == "age":
        scale = False
        xaxis = "Days"
        filename = "day"
    elif columns == "novelty":
        scale = False
        xaxis = "Novelty score"
        filename = "novelty"
    elif columns == "node_count":
        scale = True
        xaxis = "Nodes"
        filename = "node"
    else:
        scale = True
        xaxis = "Transactions"
        filename = "transaction"
    TGS_stats_df = pd.read_csv('../TGS_stats.csv')

    test_data = file_to_array("../../data/data_package/datasets_testing_package.txt")

    test_stats = TGS_stats_df[TGS_stats_df['dataset'].isin(test_data)][columns].tolist()

    # print
    plt.figure(figsize=(10, 9))  # Width, height in inches

    g = sns.histplot(test_stats, log_scale=scale, bins=20)
    g.set_xlabel(xaxis, fontsize=30)
    g.set_ylabel('Frequency', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=20)

    plt.savefig("../../pic/test_{}_TGS.pdf".format(filename))

    plt.show()


# print(train_data)

def random_rgb_color():
    """Generate a random RGB color."""
    return tuple(np.random.randint(0, 256, size=3) / 255)


def draw_toper_3D(list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for df in list:
        x = df['x'].tolist()
        y = df['y'].tolist()
        z = df['t'].tolist()

        sns.set(style="whitegrid")

        # Plot the points
        scatter = ax.plot(x, y, z, c=random_rgb_color(), alpha=0.6, marker='o')

        # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

def draw_distribution(df_TGS_stats):
    with open('performace_split.json', 'r') as json_file:
        data = json.load(json_file)

    single_dataset = data['single']
    foundation_dataset = data['foundation']

    single = []
    foundation = []
    column = 'transitivity'
    name = 'transitivity'

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


if __name__ == '__main__':
    df_single = pd.read_csv("../../data/output/single_model_train_set/results.csv")
    df_foundation = pd.read_csv("../../data/output/fm-64-results.csv")
    datasets_list = pd.read_csv("../../data/data_package/datasets_package_64.txt").iloc[:, 0].values
    df_TGS = pd.read_csv("../../data/TGS_available_datasets.csv")
    df_baseline = pd.read_csv("../../data/output/baselinemodel.csv")
    df_TGS_stats = pd.read_csv("../../data/TGS_stats.csv")

    draw_distribution(df_TGS_stats)
