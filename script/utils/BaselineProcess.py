import os
import shutil
import datetime as dt
import pandas as pd
#
root_path = "../../data/input/tokens/"
timeseries_file_path = "../../data/input/tokens/"


def creatBaselineDatasets(file,root,save_root, normalization=False):
    save_file_name = file.split(".")[0].replace("_", "")
    label_file_path =  "{}/labels/{}_labels.csv".format(save_root,save_file_name)
    edgelist_file_path = "{}/edgelists/{}_edgelist.txt".format(save_root,save_file_name)

    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0
    batch_size = 20
    batch_lables = []

    if os.path.exists(label_file_path):
        os.remove(label_file_path)

    if os.path.exists(edgelist_file_path):
        os.remove(edgelist_file_path)


    appended_edgelist_df = pd.DataFrame()

    selectedNetwork = pd.read_csv((root + file), sep=',')
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()

    # drop unused data
    selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
    selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
    selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
    selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

    # rename columns
    selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

    print(f"{file} -- {window_start_date} -- {data_last_date}")

    print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
    # check if the network has more than 20 days of data
    if ((data_last_date - window_start_date).days < minValidDuration):
        print(file + "Is not a valid network")
        # shutil.move(root_path + file, root_path + "Invalid/" + file)
        return

    # normalize the edge weights for the graph network {0-9}
    max_transfer = float(selectedNetwork['weight'].max())
    min_transfer = float(selectedNetwork['weight'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # value normalization
    if normalization:
        selectedNetwork['weight'] = selectedNetwork['weight'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    # Graph Generation Process and Labeling

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
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
        selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)

        appended_edgelist_df = appended_edgelist_df._append(selectedNetworkInGraphDataWindow, ignore_index=True)

        # ------------------------------------------------
        # Storing each snapshot label data

        batch_lables.append(label)
        # Open a file in append mode and write a line to it
        if (indx % batch_size == 0):
            appended_edgelist_df.to_csv(edgelist_file_path, index=False)

            with open(label_file_path, 'a') as file_label:
                for l in batch_lables:
                    file_label.write(str(l) + "\n")
                file_label.close()

            batch_lables = []
            # print("Caching step done for batch {}".format(indx / 20))
        # --------------------------------------------------
        # print("Snapshot {} Done ".format(indx))

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed! 100%")

def createDummyBaselineLabelsFisrtDayLastDay(file, normalization=False):
    save_file_name = file.split(".")[0].replace("_", "")
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0
    batch_size = 20
    batch_lables = []


    selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=',')
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()

    # drop unused data
    selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
    selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
    selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
    selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

    # rename columns
    selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

    print(f"{file} -- {window_start_date} -- {data_last_date}")

    print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
    # check if the network has more than 20 days of data
    if ((data_last_date - window_start_date).days < minValidDuration):
        print(file + "Is not a valid network")
        shutil.move(root_path + file, root_path + "Invalid/" + file)
        return

    # normalize the edge weights for the graph network {0-9}
    max_transfer = float(selectedNetwork['weight'].max())
    min_transfer = float(selectedNetwork['weight'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # value normalization
    if normalization:
        selectedNetwork['weight'] = selectedNetwork['weight'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    base_progress = (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)
    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nCompleted Process  {} % ".format(

            (1 - ((data_last_date - window_start_date).days / (
                        windowSize + gap + lableWindowSize)) / base_progress) * 100))
        indx += 1

        # select window data
        window_end_date = window_start_date + dt.timedelta(days=windowSize)
        selectedNetworkInGraphDataWindowFirstDate = selectedNetwork[
            (selectedNetwork['date'] >= window_start_date) & (
                    selectedNetwork['date'] < window_start_date + dt.timedelta(days=1))]

        selectedNetworkInGraphDataWindowLastDate = selectedNetwork[
            (selectedNetwork['date'] >= window_end_date - dt.timedelta(days=1)) & (
                    selectedNetwork['date'] < window_end_date)]

        # generating the label for this window
        # 1 -> Increading Transactions 0 -> Decreasing Transactions
        label = 1 if (len(selectedNetworkInGraphDataWindowLastDate) - len(
            selectedNetworkInGraphDataWindowFirstDate)) > 0 else 0

        # Storing the new snapshot data after processing

        # ------------------------------------------------
        # Storing each snapshot label data
        label_file_path = "../data/input/raw/labels/" + save_file_name + "_dummy_fd_ld_labels.csv"
        batch_lables.append(label)
        # Open a file in append mode and write a line to it
        if (indx % batch_size == 0):
            with open(label_file_path, 'a') as file_label:
                for l in batch_lables:
                    file_label.write(str(l) + "\n")
                file_label.close()

            batch_lables = []
            # print("Caching step done for batch {}".format(indx / 20))
        # --------------------------------------------------
        # print("Snapshot {} Done ".format(indx))

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed! 100%")


def createDummyBaselineLabelsWeekly(file, normalization=False):
    save_file_name = file.split(".")[0].replace("_", "")
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0
    batch_size = 20
    batch_lables = []


    selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=',')
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()

    # drop unused data
    selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
    selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
    selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
    selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

    # rename columns
    selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

    print(f"{file} -- {window_start_date} -- {data_last_date}")

    print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
    # check if the network has more than 20 days of data
    if ((data_last_date - window_start_date).days < minValidDuration):
        print(file + "Is not a valid network")
        shutil.move(root_path + file, root_path + "Invalid/" + file)
        return

    # normalize the edge weights for the graph network {0-9}
    max_transfer = float(selectedNetwork['weight'].max())
    min_transfer = float(selectedNetwork['weight'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # value normalization
    if normalization:
        selectedNetwork['weight'] = selectedNetwork['weight'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    base_progress = (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        # print("\nCompleted Process  {} % ".format(
        #
        #     (1 - ((data_last_date - window_start_date).days / (
        #                 windowSize + gap + lableWindowSize)) / base_progress) * 100))
        indx += 1

        window_end_date = window_start_date + dt.timedelta(days=windowSize)
        selectedNetworkInGraphDataWindow = selectedNetwork[
            (selectedNetwork['date'] >= window_start_date) & (
                    selectedNetwork['date'] < window_end_date)]

        # select labeling data
        label_end_date = window_start_date
        label_start_date = window_start_date - dt.timedelta(days=windowSize)
        selectedNetworkInLbelingWindow = selectedNetwork[
            (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]



        # generating the label for this window
        # 1 -> Increading Transactions 0 -> Decreasing Transactions
        label = 1 if (len(selectedNetworkInLbelingWindow) - len(
            selectedNetworkInGraphDataWindow)) > 0 else 0

        # Storing the new snapshot data after processing

        # ------------------------------------------------
        # Storing each snapshot label data
        label_file_path = "../data/input/raw/labels/" + save_file_name + "_dummy_fd_ld_labels.csv"
        batch_lables.append(label)
        # Open a file in append mode and write a line to it
        if (indx % batch_size == 0):
            with open(label_file_path, 'a') as file_label:
                for l in batch_lables:
                    file_label.write(str(l) + "\n")
                file_label.close()

            batch_lables = []
            # print("Caching step done for batch {}".format(indx / 20))
        # --------------------------------------------------
        # print("Snapshot {} Done ".format(indx))

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed! 100%")
#
# datasets_list_name = [
#         # 'unnamed_token_1898_0x00a8b738e453ffd858a7edf03bccfe20412f0eb0.csv'
#         'unnamed_token_21624_0x83e6f1e41cdd28eaceb20cb649155049fac3d5aa.csv',
#         'unnamed_token_21630_0xcc4304a31d09258b0029ea7fe63d032f52e44efe.csv',
#         'unnamed_token_21635_0xe53ec727dbdeb9e2d5456c3be40cff031ab40a55.csv',
#         'unnamed_token_21636_0xfca59cd816ab1ead66534d82bc21e7515ce441cf.csv',
#         'unnamed_token_21639_0x1ceb5cb57c4d4e2b2433641b95dd330a33185a44.csv',
#         'unnamed_token_21654_0x09a3ecafa817268f77be1283176b946c4ff2e608.csv',
#         'unnamed_token_21655_0xbcca60bb61934080951369a648fb03df4f96263c.csv',
#         'unnamed_token_21658_0x5f98805a4e8be255a32880fdec7f6728c6568ba0.csv',
#         'unnamed_token_21662_0x429881672b9ae42b8eba0e26cd9c73711b891ca5.csv'
#     ]
# for data in datasets_list_name:
#     createDummyBaselineLabelsWeekly(data)
#
#
if __name__ == '__main__':
    dataset_df = pd.read_csv('dataset_no_gap_1_day.csv')

    filtered_df = dataset_df[(dataset_df['networkSize'] <=70 )]
    print(max(dataset_df['networkSize'].tolist()))
    # filtered_df.to_csv('dataset_no_gap_1_day_lt_100',index=False)
    # dataset_name = filtered_df['filename'].unique().tolist()
    # for dataset in dataset_name[19:]:
    #
    #     try:
    #         creatBaselineDatasets(dataset,"E:/token/",'../../data/data_gt_70/')
    #     except Exception as e:
    #         print("ERROR: Can't process {}\n{}".format(dataset,e))