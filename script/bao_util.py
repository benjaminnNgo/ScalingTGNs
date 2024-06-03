import math
import os
import statistics
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import shutil

#Temp
# new_datas = os.listdir('./data/input/cached/')
# print(len(new_datas))
#
# old_datas = os.listdir('../data/input/cached/')
#
# counter = 0
# for new_data in new_datas:
#     if new_data not in old_datas:
#         counter += 1
# print(counter)
#
# with open("../data/data_package/updated_data.txt", "w") as file:
#     for new_data in new_datas:
#         file.write("{}\n".format(new_data))

#Move file and remove _



# TGS_datasets = pd.read_csv("../data/TGS_available_datasets.csv")
# # old_root = "E:/token/"
# new_root = "E:/TGS/"
# available_datasets = os.listdir(new_root)
# # os.makedirs(new_root, exist_ok=True)
# counter = 0
#
# for filename in TGS_datasets['filename'].tolist():
#     if filename not in available_datasets:
#         counter += 1
#
# print(counter)

# print(len(os.listdir(new_root)))
#
# TGS_datasets['filename'] = TGS_datasets['filename'].str.replace("_","")
# TGS_datasets.to_csv("TGS_available_datasets.csv",index=False)




#=================Count number of nodes=============
def TGS_node_counts_distribution():
    TGS_datasets = pd.read_csv("../data/TGS_available_datasets.csv")
    token_root = 'E:/TGS/'

    #cache
    node_counts = np.array(
        [117423, 11325, 523632, 65350, 86895, 543220, 288659, 113453, 581396, 105725, 105738, 35156, 332643, 221373,
         8556, 63079, 2706788, 240487, 109935, 1392275, 344344, 117977, 83282, 23208, 107140, 380634, 435821, 51355,
         88247, 71667, 28033, 23289, 268450, 163408, 19079, 4893910, 4800, 63042, 12985, 586115, 277002, 1250413,
         198177, 340455, 468075, 487490, 449986, 318573, 222402, 474769, 60562, 285005, 391608, 129967, 63428, 348497,
         288318, 293409, 134338, 360609, 294502, 34680, 50415, 23689, 75616, 94156, 48241, 128752, 46738, 359916,
         119984, 39408, 128159, 276319, 69230, 83299, 87186, 39323, 34328, 145859, 79984, 23742, 21430, 25852, 26799,
         28498, 48685, 47046, 43964, 71590, 34687, 118294, 91218, 45728, 16874, 52586, 52753, 34051, 60472, 220476,
         75696, 54885, 28382, 59785, 49723, 46087, 23038, 83713, 32754, 83478, 25248, 31509, 45645, 31734, 20488, 26179,
         53385, 14590, 40627, 45342, 22466, 42806, 55330, 35277, 42539, 19948, 50645, 34341, 36517, 37486, 48458, 11008,
         35881, 38638, 39061, 67599, 17282, 21116, 29798, 25, 30931, 19063, 24514, 13146, 26221, 20346, 11647, 21531,
         25450, 108714, 27201, 60717, 24185, 33728, 42592, 22574, 30721, 26153, 45920, 14378, 31965, 23793, 44733,
         13193, 29201, 14299, 40023, 110570, 26000, 6765, 34704, 18420, 11222, 8730, 18692, 9992, 44597, 14472, 14782,
         30737, 101699, 14567, 11860, 21757, 13658, 26521, 29231, 40476, 19990, 27245, 24277, 37355, 13669, 14924, 8450,
         9127, 20085, 15784, 14686, 9426, 26867, 7471, 37004, 12137, 12091, 16355, 14013, 17962, 12324, 89140, 25344,
         19512, 18401, 8430, 8497, 9883, 15837, 1454, 7854, 14501, 25852, 12838, 6675, 10228, 2950, 8969, 5949, 14238,
         12401, 7943, 14791, 22398, 20792, 12426, 12057, 22441, 15089, 7552, 15388, 5075, 16037, 8483, 7664, 2155,
         15724, 9294, 21035, 4946, 9216, 8082, 14457, 10662, 11331, 5291, 9202, 13026, 15738, 11346, 12076, 12365, 6494,
         6376, 11819, 2976, 10414, 31864, 6947, 202675, 7402, 44010, 13648, 40055, 64261, 9205]
    )
    node_counts = np.log10(node_counts)

    # for dataset in TGS_datasets['filename'].tolist():
    #     # print(dataset)
    #     data_df = pd.read_csv("{}{}".format(token_root,dataset))
    #     unique_nodes = set()
    #     unique_nodes.update(data_df['from'].tolist())
    #     unique_nodes.update(data_df['to'].tolist())
    #     node_counts.append(len(unique_nodes))

    # print(node_counts)
    sns.histplot(node_counts, bins=50, edgecolor='black')
    # Add labels and title
    plt.xlabel('Number of nodes (log base 10)')
    plt.ylabel('Frequency')
    plt.title('Number of unique nodes distribution')
    plt.show()

def TGS_duration_distribution():
    TGS_datasets = pd.read_csv("../data/TGS_available_datasets.csv")
    token_root = 'E:/TGS/'
    durations_distribution = []
    for dataset in TGS_datasets['filename'].tolist():
        # print(dataset)
        data_df = pd.read_csv("{}{}".format(token_root,dataset))
        data_df['date'] = pd.to_datetime(data_df['timestamp'], unit='s').dt.date
        window_start_date = data_df['date'].min()
        data_last_date = data_df['date'].max()
        duration =(data_last_date - window_start_date).days
        durations_distribution.append(duration)
    print(durations_distribution)
    sns.histplot(durations_distribution, bins=50, edgecolor='black')
    # Add labels and title
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Age distribution')
    plt.show()

def TGS_node_transaction_time_distribution():
    TGS_datasets = pd.read_csv("../data/TGS_available_datasets.csv")
    token_root = 'E:/TGS/'
    rowlist = []
    columns = ['dataset','node_count','transaction_count','age']

    for dataset in TGS_datasets['filename'].tolist():
        data_df = pd.read_csv("{}{}".format(token_root, dataset))
        row = []
        row.append(dataset)

        #node_count
        unique_nodes = set()
        unique_nodes.update(data_df['from'].tolist())
        unique_nodes.update(data_df['to'].tolist())
        row.append(len(unique_nodes))

        #Transaction count
        transaction_count = data_df.shape[0]
        row.append(transaction_count)

        #age
        data_df['date'] = pd.to_datetime(data_df['timestamp'], unit='s').dt.date
        window_start_date = data_df['date'].min()
        data_last_date = data_df['date'].max()
        duration = (data_last_date - window_start_date).days
        row.append(duration)

        rowlist.append(row)

    pd.DataFrame(rowlist,columns=columns).to_csv("TGS_stats.csv",index=False)

def TGS_transaction_count_distribution():
    TGS_datasets = pd.read_csv("../data/TGS_available_datasets.csv")
    token_root = 'E:/TGS/'
    transaction_distribution = []
    for dataset in TGS_datasets['filename'].tolist():
        # print(dataset)
        data_df = pd.read_csv("{}{}".format(token_root, dataset))
        transaction_count = data_df.shape[0]
        transaction_distribution.append(transaction_count)
    print(transaction_distribution)
    sns.histplot(transaction_distribution, bins=50, edgecolor='black')
    # Add labels and title
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Age distribution')
    plt.show()

def plot_hist(data_df,columns,title,log = False,kde = True):
    target_count = np.array(data_df[columns].tolist())
    # node_count = np.log10(node_count)
    if not log:
        g = sns.histplot(target_count, bins=50, kde=kde)
    else:
        g = sns.histplot(target_count, bins=50, kde=kde,log_scale=True)
    # g.set(xscale="log")
    g.set_xlabel(columns)
    g.set_ylabel('Frequency')
    plt.title(title)
    plt.show()


def plot_hist_from_list(list,title,log = False,kde = True):
    target_count = np.array(list)
    # node_count = np.log10(node_count)
    if not log:
        g = sns.histplot(target_count, bins=50, kde=kde)
    else:
        g = sns.histplot(target_count, bins=50, kde=kde,log_scale=True)
    # g.set(xscale="log")
    g.set_xlabel(list)
    g.set_ylabel('Frequency')
    plt.title(title)
    plt.show()

def get_edge_train_and_test(data_df,train_ratio = 0.7, test_ratio = 0.15):
    edges_train = set()
    edges_test = set()

    number_snapshot = int(data_df['snapshot'].max())
    snapshots = list(range(1,number_snapshot + 1))
    training_shot_number = round(number_snapshot*train_ratio)
    test_shot_number = round(number_snapshot * test_ratio)

    train_snapshots = snapshots[:training_shot_number]
    test_snapshots = snapshots[-test_shot_number:]

    for idx_snapshot in train_snapshots:
        TG_snapshot = data_df[data_df['snapshot'] == idx_snapshot]
        for index,row in TG_snapshot.iterrows():
            edges_train.add((str(row['source']),str(row['destination'])))

    for idx_snapshot in test_snapshots:
        TG_snapshot = data_df[data_df['snapshot'] == idx_snapshot]
        for index,row in TG_snapshot.iterrows():
            edges_test.add((str(row['source']),str(row['destination'])))

    return edges_train,edges_test
def calc_reocurrence(data_df):
    edges_train,edges_test = get_edge_train_and_test(data_df)
    reocurrence_set = edges_train & edges_test

    return len(reocurrence_set)/len(edges_train)

def calc_surprise(data_df):
    edges_train,edges_test = get_edge_train_and_test(data_df)
    different_set = edges_test.difference(edges_train)

    return len(different_set)/len(edges_test)



def calc_novelty(data_df):
    data_df['date'] = pd.to_datetime(data_df['timestamp'], unit='s').dt.date
    unique_dates = list(data_df['date'].unique())
    unique_dates.sort()
    seen_edge = set()
    ratios = []
    for timestamp in unique_dates:
        graph_timestamp = data_df[data_df['date'] == timestamp]
        t_edge = set()
        for index,row in graph_timestamp.iterrows():
            t_edge.add((str(row['from']), str(row['to'])))

        never_seen_edge = t_edge.difference(seen_edge)
        ratio = len(never_seen_edge)/len(t_edge)
        ratios.append(ratio)
        seen_edge.update(t_edge)
    return sum(ratios)/len(ratios)

def check_valid_dataset(labels):
    unique_labels = set(labels)
    return len(unique_labels) != 1

def find_label_csv(dataset):
    dataset = dataset.replace('.csv', '')
    partial_path = '../data/'
    potiential_packages = ['data_bw_25_and_40/', 'data_bw_40_and_70/', 'data_gt_70/', 'data_lt_25MB/', 'input/raw/']
    for package in potiential_packages:
        file_path = "{}{}/labels/{}_labels.csv".format(partial_path, package, dataset)
        if os.path.exists(file_path):
            return file_path
    raise Exception("Can't find label file")

def find_edge_csv(dataset):
    dataset = dataset.replace('.csv', '')
    partial_path = '../data/'
    potiential_packages = ['data_bw_25_and_40/', 'data_bw_40_and_70/', 'data_gt_70/', 'data_lt_25MB/', 'input/raw/']
    for package in potiential_packages:
        file_path = "{}{}/edgelists/{}_edgelist.txt".format(partial_path, package, dataset)
        if os.path.exists(file_path):
            return file_path
    raise Exception("Can't find label file{}".format(file_path))

def get_val_test(labels,test_ratio = 0.15, val_ratio = 0.15):
    total_snapshot = len(labels)
    test_and_validation_total = round(total_snapshot * (test_ratio + val_ratio))
    test_and_validation_snapshot =labels[-test_and_validation_total:]

    test_total = round(total_snapshot * test_ratio)
    val_total = round(total_snapshot * val_ratio)

    return test_and_validation_snapshot[:val_total], test_and_validation_snapshot[-test_total:]

def compute_novelty_dist_from_datasets(datasets):
    novelty_list = []
    for dataset in datasets:
        print("Compute novelty score for {}".format(dataset))
        pd_df = pd.read_csv('E:/TGS/{}.csv'.format(dataset))
        novelty_list.append(calc_novelty(pd_df))



    sns.histplot(novelty_list, bins=50, edgecolor='black')
    # Add labels and title
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Age distribution')
    plt.show()
    return novelty_list

def compute_reocurrence_surprise_from_datasets(datasets):
    reoccurrence_list = []
    surprise_list = []
    for dataset in datasets:
        dataset_df = pd.read_csv(find_edge_csv(dataset))
        reoccurrence_list.append(calc_reocurrence(dataset_df))
        surprise_list.append(calc_surprise(dataset_df))
    return reoccurrence_list, surprise_list



if __name__ == '__main__':
    # Load the example planets dataset
    # df = pd.read_csv('E:/TGS/unnamedtoken219740xcafe001067cdef266afb7eb5a286dcfd277f3de5.csv')
    # print(calc_novelty(df))



    # # TGS_node_transaction_time_distribution()
    # dataset_in_package_df = pd.read_csv('../data/data_package/datasets_package_64.txt')
    # dataset_64 = dataset_in_package_df.iloc[:,0].tolist()
    # # print(dataset_64)
    #
    # partial_path = '../data/'
    # potiential_packages = ['data_bw_25_and_40/','data_bw_40_and_70/','data_gt_70/','data_lt_25MB/','input/raw/']
    # cant_find_dataset = []
    # invalid_dataset = []
    #
    # for dataset in dataset_64:
    #     try:
    #         label_path = find_label_csv(dataset)
    #         labels = pd.read_csv(label_path).iloc[:,0].tolist()
    #         val_set,test_set = get_val_test(labels)
    #         if not check_valid_dataset(val_set) and not check_valid_dataset(test_set):
    #             invalid_dataset.append(dataset)
    #
    #     except Exception as e:
    #         cant_find_dataset.append(cant_find_dataset)
    #
    #
    # print(invalid_dataset)
    # set: 9[
    #     'unnamedtoken220260x20561172f791f915323241e885b4f7d5187c36e1'

    #     , 'unnamedtoken221770x511686014f39f487e5cdd5c37b4b37606b795ae3'

    #     , 'unnamedtoken219600x75c97384ca209f915381755c582ec0e2ce88c1ba'

    #     , 'unnamedtoken222140xe0f63a424a4439cbe457d80e4f4b51ad25b2c56c'

    #     , 'unnamedtoken218200xa62894d5196bc44e4c3978400ad07e7b30352372'

    #     , 'unnamedtoken221880x0b0a8c7c34374c1d0c649917a97eee6c6c929b1b'

    #     , 'unnamedtoken219650xc50c1c8b7cfcc868cae13654134f1078b3b8a0f2'

    #     , 'unnamedtoken221890xcb50350ab555ed5d56265e096288536e8cac41eb'

    #     , 'unnamedtoken214030x07e0edf8ce600fb51d44f51e3348d77d67f298ae'

    #        unnamedtoken221880x0b0a8c7c34374c1d0c649917a97eee6c6c929b1b
    #        unnamedtoken221870x3e34eabf5858a126cb583107e643080cee20ca64
    #     ]

    # print("invalid set:",len(invalid_dataset),invalid_dataset)
    # print("can't find set",cant_find_dataset)

    # with open('filter_available_test.txt', 'w') as file:
    #     # Write each element of the array to the file
    #     for element in valid_dataset:
    #         file.write(str(element) + '\n')

    # TGS_available_dataset = pd.read_csv('../data/TGS_available_datasets.csv')['filename'].tolist()

    # for dataset in TGS_available_dataset:
    #     pd_df = pd.read_csv('E:/TGS/{}'.format(dataset))
    #     novelty_list.append(calc_novelty(pd_df))

    #
    # # print(novelty_list)
    # sns.histplot(novelty_list, bins=50, edgecolor='black')
    # # Add labels and title
    # plt.xlabel('Days')
    # plt.ylabel('Frequency')
    # plt.title('Age distribution')
    # plt.show()

    # with open('../data/data_package/datasets_package_64.txt', 'r') as file:
    #     # Read all lines into a list
    #     lines = file.readlines()
    #
    # # Strip newline characters and create an array
    # dataset_64 = set([line.strip() for line in lines])
    #
    # with open('../data/data_package/datasets_package_32.txt', 'r') as file:
    #     # Read all lines into a list
    #     lines = file.readlines()
    #
    #     # Strip newline characters and create an array
    # dataset_32 = set([line.strip() for line in lines])
    #
    # with open('../data/data_package/datasets_package_16.txt', 'r') as file:
    #     # Read all lines into a list
    #     lines = file.readlines()
    #
    #     # Strip newline characters and create an array
    # dataset_16 = set([line.strip() for line in lines])
    #
    # with open('../data/data_package/datasets_package_8.txt', 'r') as file:
    #     # Read all lines into a list
    #     lines = file.readlines()
    #
    #     # Strip newline characters and create an array
    # dataset_8 = set([line.strip() for line in lines])


    # print(len(dataset_32.intersection(dataset_32)))
    # print(len(dataset_32.intersection(dataset_16)))
    # print(len(dataset_32.intersection(dataset_8)))

    # not_used_data = []
    # # dataset_in_package_df = pd.read_csv('')
    # for data in os.listdir("../data/input/cached/"):
    #     if data not in dataset_64 and data not in dataset_32 and data not in dataset_16:
    #         not_used_data.append(data)
    #
    #
    # not_used_data.remove("temp")
    # print(len(not_used_data))
    #
    # partial_path = '../data/'
    # potiential_packages = ['data_bw_25_and_40/','data_bw_40_and_70/','data_gt_70/','data_lt_25MB/','input/raw/']
    # cant_find_dataset = []
    # valid_data = set()
    #
    # for dataset in not_used_data:
    #     try:
    #         label_path = find_label_csv(dataset)
    #         labels = pd.read_csv(label_path).iloc[:,0].tolist()
    #         val_set,test_set = get_val_test(labels)
    #         if check_valid_dataset(val_set) and check_valid_dataset(test_set):
    #             valid_data.add(dataset)
    #
    #     except Exception as e:
    #         cant_find_dataset.append(cant_find_dataset)
    #
    # for dataset in valid_data:
    #     print(dataset)

    # with open('../data/data_package/datasets_testing_package.txt', 'r') as file:
    #     # Read all lines into a list
    #     lines = file.readlines()
    #
    #     # Strip newline characters and create an array
    # test_data = set([line.strip() for line in lines])
    # print(len(test_data.intersection(dataset_64)))

    # partial_path = '../data/'
    # potiential_packages = ['data_bw_25_and_40/','data_bw_40_and_70/','data_gt_70/','data_lt_25MB/','input/raw/']
    # cant_find_dataset = []
    # invalid_data = set()
    #
    # for dataset in not_used_data:
    #     try:
    #         label_path = find_label_csv(dataset)
    #         labels = pd.read_csv(label_path).iloc[:,0].tolist()
    #         val_set,test_set = get_val_test(labels)
    #         if not check_valid_dataset(val_set) or not check_valid_dataset(test_set):
    #             invalid_data.add(dataset)
    #
    #     except Exception as e:
    #         cant_find_dataset.append(cant_find_dataset)
    # print(invalid_data)

    # datadf = pd.read_csv("TGS_stats.csv")
    # plot_hist(datadf,'age','Age distribution')


#move file in flash drive
    # import os
    # import shutil
    #
    # new_place = "D:/TGS"
    # source = "E:/TGS/"
    # counter = 0
    #
    #
    # for file in os.listdir(source):
    #     if counter >= 5:
    #         break;
    #     dataname = file.replace(".csv","")
    #     destination = "{}/{}/".format(new_place,dataname)
    #     os.makedirs(destination, exist_ok=True)
    #
    #     try:
    #         shutil.copy(source + file, destination)
    #         print(f"File '{source}' successfully copied to '{destination}'.")
    #     except FileNotFoundError:
    #         print(f"The source file '{source + file}' does not exist.")
    #     except PermissionError:
    #         print(f"Permission denied. Could not copy '{source + file}' to '{destination}'.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #
    #     counter+=1
    #     print(dataname)


#Refilter TGS datasets
    # TGS_datasets_list = pd.read_csv("../data/TGS_available_datasets.csv")
    # TGS_stats_df = pd.read_csv("TGS_stats_old.csv")
    # TGS_datasets_list= TGS_datasets_list['dataset'].tolist()
    # TGS_stats_df['dataset'] = TGS_stats_df['dataset'].str.replace(".csv","",regex=False)
    # TGS_stats_df = TGS_stats_df[TGS_stats_df['dataset'].isin(TGS_datasets_list)]
    # TGS_stats_df.to_csv("TGS_stats.csv",index=False)
    # print(TGS_stats_df)

#Compute label rate
    # TGS_stats_df = pd.read_csv("TGS_stats.csv")
    # labels_rate_list = []
    # for dataset in TGS_stats_df['dataset'].tolist():
    #     # print(dataset)
    #     labels_df = pd.read_csv('../data/input/raw/labels/{}_labels.csv'.format(dataset))
    #     list_labels = labels_df.iloc[:,0].tolist()
    #     labels_rate_list.append( sum(list_labels)/len(list_labels))
    #
    # TGS_stats_df['label_1_rate']= labels_rate_list
    # TGS_stats_df.to_csv("TGS_stats.csv")
    # print(TGS_stats_df)
    # print(sum(labels_rate_list)/len(labels_rate_list))

#Compute novelty
    # TGS_stats_df = pd.read_csv("TGS_stats.csv")
    # novelty_score_list = compute_novelty_dist_from_datasets(TGS_stats_df['dataset'].tolist())
    # print(novelty_score_list)
    #
    #
    # TGS_stats_df['novelty']= novelty_score_list
    # TGS_stats_df.to_csv("TGS_stats.csv")

#Drop column

    # TGS_stats_df = pd.read_csv("TGS_stats.csv")
    # TGS_stats_df = TGS_stats_df.drop(TGS_stats_df.columns[0], axis=1)
    # TGS_stats_df.to_csv("TGS_stats.csv",index= False)
    #
    # print(TGS_stats_df)

#Compute suprise score
    # TGS_stats_df = pd.read_csv("TGS_stats.csv")
    # surprise_list = []
    # for dataset in TGS_stats_df['dataset'].tolist():
    #     print(dataset)
    #     edge_df = pd.read_csv(find_edge_csv(dataset))
    #     surprise = calc_surprise(edge_df)
    #     surprise_list.append(surprise)
    #     # labels_df = pd.read_csv('../data/input/raw/labels/{}_labels.csv'.format(dataset))
    #     # list_labels = labels_df.iloc[:,0].tolist()
    #     # labels_rate_list.append( sum(list_labels)/len(list_labels))
    # print(surprise_list)
    # TGS_stats_df['surprise']= surprise_list
    # TGS_stats_df.to_csv("TGS_stats.csv")
    # print(TGS_stats_df)
    # print(sum(labels_rate_list)/len(labels_rate_list))

#Move from TGS folder to TGS_official

    # import os
    # import shutil
    # TGS_datalist = pd.read_csv("../data/TGS_available_datasets.csv")
    # new_place = "E:/TGS_official/"
    # source = "E:/TGS/"
    # counter = 0
    #
    #
    # for index,row in TGS_datalist.iterrows():
    #     filename = row['dataset'] + ".csv"
    #     destination = "{}/{}.csv".format(new_place,row['token_name'])
    #     # os.makedirs(destination, exist_ok=True)
    #
    #     try:
    #         shutil.copy(source + filename, destination)
    #         print(f"File '{source}' successfully copied to '{destination}'.")
    #     except FileNotFoundError:
    #         print(f"The source file '{source + filename}' does not exist.")
    #     except PermissionError:
    #         print(f"Permission denied. Could not copy '{source + filename}' to '{destination}'.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #
    #     counter+=1
    #     # print(filename)
    # print(len(TGS_datalist['token_name'].tolist()))
    # unique_tokens = set()
    # for dataset in TGS_datalist['token_name'].tolist():
    #     if dataset not in unique_tokens:
    #         unique_tokens.add(dataset)
    #     else:
    #         print(dataset)
    #
    #
    # print(len(os.listdir("E:/TGS_official/")))
    # for index,row in TGS_datalist.iterrows():
    #     destination = "{}/{}.csv".format(new_place, row['token_name'])
    #     if not os.path.exists(destination):
    #         print("Can't find {}".format(destination))

#Check smallest date
    # lastest_day =date(2010, 5, 29)
    #
    # for filename in os.listdir("E:/TGS_official/"):
    #     selectedNetwork = pd.read_csv("E:/TGS_official/" +filename, sep=',')
    #     selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
    #     selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    #     selectedNetwork = selectedNetwork.sort_values(by='date')
    #     window_start_date = selectedNetwork['date'].max()
    #     if window_start_date > lastest_day:
    #         smallest_day = window_start_date
    #
    # print(smallest_day)

# Generate table
    TGS_availabe_df = pd.read_csv("../data/TGS_available_datasets.csv")
    TGS_stats_df = pd.read_csv("TGS_stats.csv")

    for index, row in TGS_stats_df.iterrows():
        token_name_row = TGS_availabe_df[TGS_availabe_df['dataset'] == row['dataset'] ]

        token_name = str(token_name_row['token_name'].values[0])
        # print(token_name)
        node_count = int(row['node_count'])
        trans_count = int(row['transaction_count'])
        age = int(row['age'])
        label_ratio = round(row['label_1_rate'],2)
        novelty, surprise = round(row['novelty'],2), round(row['surprise'],2)

        prepare_string = "{} & {} & {} & {} & {} & {} & {}\\\\".format(token_name,
                                                                       node_count,
                                                                       trans_count,
                                                                       age,
                                                                       label_ratio,
                                                                       novelty,
                                                                       surprise)
        print(prepare_string)







