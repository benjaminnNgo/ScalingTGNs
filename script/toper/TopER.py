from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmean import find_optimal_k
# General function
np.random.seed(702)
def best_fit_rep(X, Y):
    n = len(X)
    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    for i in range(n):
        A = A + 1
        B = B + X[i]
        C = C + X[i] * X[i]
        D = D + X[i] * Y[i]
        E = E + Y[i]

    a = (C * E - B * D) / (A * C - B * B)
    b = (-B * E + A * D) / (A * C - B * B)
    M = [a, b]

    return M



def TopER(G, threshold_array, num_chunks):
    '''
      G: nx graph
      num_chunk: the max number of thresholds you want to use
      '''


    if len(threshold_array) <= num_chunks:
        thresh = threshold_array
    elif len(threshold_array) > num_chunks:
        chunk_size = len(threshold_array) // num_chunks
        remainder = len(threshold_array) % num_chunks

        # Divide the list into num_chunks chunks
        chunks = [threshold_array[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)]
                  for i in range(num_chunks)]

        thresh = []
        for i in range(num_chunks):
            if i == 0:
                thresh.append(min(chunks[0]))
            thresh.append(max(chunks[i]))



    c0 = []
    c1 = []
    for val in thresh:
        selected_edges = [(u, v, w) for u, v, w in G.edges(data='weight') if w <= val]

        if len(selected_edges) > 0:
            sub = G.edge_subgraph(selected_edges)
            c0.append(sub.number_of_nodes())
            c1.append(len(selected_edges))
        else:
            c0.append(0)
            c1.append(0)

    M = best_fit_rep(c0, c1)
    return M

def create_subgragh_from_pack(dataset,normalization = False, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    list_G = []
    partial_path = "../../data/input/raw"
    # partial_path = "../../data/input/raw"


    # load and process graph-pooled (node-level) features
    edgelist_filename = f'{partial_path}/edgelists/{dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)



    # value normalization
    if normalization:
        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(edgelist_df['weight'].max())
        min_transfer = float(edgelist_df['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        edgelist_df['weight'] = edgelist_df['weight'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    uniq_weights_list = np.unique(edgelist_df['weight'])

    for ts in uniq_ts_list:
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight',
                                       create_using=nx.MultiDiGraph)

        list_G.append(ts_G)

    return list_G,uniq_weights_list

def generate_toper_from_list(dataset_list,normalization = False):

        for dataset in dataset_list:
            try:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("*INFO:[{}] PROCESSING DATASET {}*".format(current_time,dataset))
                list_G, uniq_weights_list = create_subgragh_from_pack(dataset,normalization=normalization)
                x = []
                y = []
                t = []
                counter = 1
                for snapshot in list_G:
                    x_coor, y_coor = TopER(snapshot, uniq_weights_list, 500)

                    x.append(x_coor)
                    y.append(y_coor)
                    t.append( counter)
                    counter+=1

                if normalization:
                    pd.DataFrame({'x': x, 'y': y,'t':t}).to_csv("./toper_values/TGS/{}_normalization.csv".format(dataset),index=False)
                else:
                    pd.DataFrame({'x': x, 'y': y,'t':t}).to_csv("./toper_values/TGS/{}.csv".format(dataset),index=False)
            except Exception as e:
                print("Can't process {}: {}".format(dataset,e))


def random_rgb_color():
    """Generate a random RGB color."""
    return tuple(np.random.randint(0, 256, size=3) / 255)

def plot_toper_values(pack_dict, kmean = False):
    plt.figure(figsize=(15,10))  # Optional: set the size of the figure
    plt.xlim(-5000, 2000)  # Set the x-axis range from 0 to 6

    for pack in pack_dict:
        dataset_list = pack_dict[pack]
        for dataset in dataset_list:
            print(dataset)
            toper_vallues_df = pd.read_csv("./toper_values/{}/{}.csv".format(pack,dataset))

            if not kmean:
                x= []
                y = []
                for index,row in toper_vallues_df.iterrows():
                    x.append(row['x'])
                    y.append(row['y'])
            else:
                points = []
                for index, row in toper_vallues_df.iterrows():
                    point = []
                    point.append(row['x'])
                    point.append(row['y'])
                    points.append(point)
                points = np.array(points)
                optimal_k = find_optimal_k(points)

                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                kmeans.fit_predict(points)
                x = kmeans.cluster_centers_[:, 0]
                y = kmeans.cluster_centers_[:, 1]

            if not kmean:
                plt.scatter(x, y, marker='o', linestyle='', color=random_rgb_color(), label=dataset.replace("unnamed",""), alpha=0.6)  # Plotting the points with markers
            else:
                plt.scatter(x, y, marker='o',linestyle='',color=random_rgb_color(),label=dataset.replace("unnamed",""), alpha=0.9,s=200)  # Plotting the points with markers

    plt.xlabel('pivot')  # Label for the x-axis
    plt.ylabel('growth')  # Label for the y-axis
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()  # Show the plot
    plt.savefig("./toper_values.png")

if __name__ == '__main__':



    TGS_data = pd.read_csv("../../data/TGS_available_datasets.csv")

    data_list = TGS_data['dataset'].tolist()
    n = len(data_list)
    size = n // 3
    list1 = data_list[:size]
    list2 = data_list[size:2 * size]
    list3 = data_list[2 * size:]
    generate_toper_from_list(list1)
    # plot_toper_values(dataset_per_pack,True)


        
    
    
