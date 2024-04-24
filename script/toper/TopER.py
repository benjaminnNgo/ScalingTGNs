import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

def create_subgragh_from_pack(dataset,pack,normalization = False, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    list_G = []
    partial_path = "../../data/{}".format(pack)

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

def generate_toper_from_package(pack_dict,normalization = False):
    for pack in pack_dict:
        print("================INFO: PROCESSING PACKAGE {}================".format(pack))
        dataset_list = pack_dict[pack]
        for dataset in dataset_list:
            print("*INFO: PROCESSING DATASET {}*".format(dataset))
            list_G, uniq_weights_list = create_subgragh_from_pack(dataset,pack,normalization=normalization)
            x = []
            y = []
            for snapshot in list_G:
                x_coor, y_coor = TopER(snapshot, uniq_weights_list, 500)

                x.append(x_coor)
                y.append(y_coor)

            if normalization:
                pd.DataFrame({'x': x, 'y': y}).to_csv("./toper_values/{}/{}_normalization.csv".format(pack,dataset),index=False)
            else:
                pd.DataFrame({'x': x, 'y': y}).to_csv("./toper_values/{}/{}.csv".format(pack,dataset),index=False)



def random_rgb_color():
    """Generate a random RGB color."""
    return tuple(np.random.randint(0, 256, size=3) / 255)

def plot_toper_values(pack_dict):
    plt.figure(figsize=(15,10))  # Optional: set the size of the figure
    plt.xlim(-1000, 1000)  # Set the x-axis range from 0 to 6

    for pack in pack_dict:
        dataset_list = pack_dict[pack]
        for dataset in dataset_list:
            toper_vallues_df = pd.read_csv("./toper_values/{}/{}.csv".format(pack,dataset))
            x = []
            y = []
            for index,row in toper_vallues_df.iterrows():
                x.append(row['x'])
                y.append(row['y'])


            plt.plot(x, y, marker='o',linestyle='',color=random_rgb_color(),label=dataset, alpha=0.5)  # Plotting the points with markers

    plt.xlabel('pivot')  # Label for the x-axis
    plt.ylabel('growth')  # Label for the y-axis
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()  # Show the plot
    plt.savefig("./toper_values.png")

if __name__ == '__main__':


    dataset_per_pack={

        "data_gt_70":[
            # 'AKITA0x3301ee63fb29f863f2333bd4466acb46cd8323e6',
            # 'BNB0xb8c77482e45f1f44de1745f52c74426c631bdd52',
            'CRO0xa0b73e1ff0b80914ab6fe0444e65848c4c34450b',
            'Geminidollar0x056fd409e1d7a124bd7017459dfea2f387b6d5cd',
            'CVC0x41e5560054824ea6b0732e656e3ad64e20e94e45'
        ],
        "data_lt_25MB":[
            'ARC0xc82e3db60a52cf7529253b4ec688f631aad9e7c2',
            # 'FNKOS0xeb021dd3e42dc6fdb6cde54d0c4a09f82a6bca29',
            # 'INU0xc76d53f988820fe70e01eccb0248b312c2f1c7ca',
            # 'unnamedtoken270x00000000051b48047be6dc0ada6de5c3de86a588',
            # 'unnamedtoken124310x04906695d6d12cf5459975d7c3c03356e4ccd460'
        ],

        "data_bw_25_and_40":[
            'CMT0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f',
            # 'CELR0x4f9254c83eb525f9fcf346490bbb3ed28a81c667',
            # 'GHST0x3f382dbd960e3a9bbceae22651e88158d2791550',
            # 'REP0xe94327d07fc17907b4db788e5adf2ed424addff6',
            'RFD0x955d5c14c8d4944da1ea7836bd44d54a8ec35ba1'
        ],
    }

    # dataset_per_pack = {
    #     "data_lt_25MB": [
    #         'ARC0xc82e3db60a52cf7529253b4ec688f631aad9e7c2',
    #         'FNKOS0xeb021dd3e42dc6fdb6cde54d0c4a09f82a6bca29',
    #         'INU0xc76d53f988820fe70e01eccb0248b312c2f1c7ca',
    #         # 'unnamedtoken270x00000000051b48047be6dc0ada6de5c3de86a588',
    #         # 'unnamedtoken124310x04906695d6d12cf5459975d7c3c03356e4ccd460'
    #         'ARC0xc82e3db60a52cf7529253b4ec688f631aad9e7c2_normalization',
    #         'FNKOS0xeb021dd3e42dc6fdb6cde54d0c4a09f82a6bca29_normalization',
    #         'INU0xc76d53f988820fe70e01eccb0248b312c2f1c7ca_normalization'
    #
    #     ],
    # }
    # generate_toper_from_package(dataset_per_pack,normalization= True)
    plot_toper_values(dataset_per_pack)


        
    
    
