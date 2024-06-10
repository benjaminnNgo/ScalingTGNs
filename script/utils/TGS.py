import os
import shutil
import datetime as dt
import pandas as pd

class TGS_Handler:
    def __init__(self, TGS_storage_path, window_size=7, gap=3, label_window_size=7, min_valid_duration=3):
        self.token_root_path = TGS_storage_path
        self.label_root_path = "../data/input/raw/labels/"
        self.dummy_label_root_path = "../data/input/raw/dummy_labels/"
        self.edgelist_path = "../data/input/raw/edgelists/"

        self.window_size = window_size  # Day
        self.gap = gap
        self.label_window_size = label_window_size  # Day
        self.min_valid_duration = min_valid_duration  # Day

        if not os.path.exists(self.label_root_path):
            os.makedirs(self.label_root_path)

        if not os.path.exists(self.edgelist_path):
            os.makedirs(self.edgelist_path)

        if not os.path.exists(self.dummy_label_root_path):
            os.makedirs(self.dummy_label_root_path)

    def create_dummy_baseline_labels_weekly(self, dataset, normalization=False):
        print("INFO: Dummy label is not yet calculating. Generating dummy labels weekly...")

        save_file_name = dataset.split(".")[0].replace("_", "")
        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        selectedNetwork = pd.read_csv((self.token_root_path + dataset + ".csv"), sep=',')
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

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            raise Exception("{} is not a valid network".format(dataset))

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        base_progress = (data_last_date - window_start_date).days / (
                    self.window_size + self.gap + self.label_window_size)

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date
            label_start_date = window_start_date - dt.timedelta(days=self.window_size)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 0 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 1

            # Storing the new snapshot data after processing

            # ------------------------------------------------
            # Storing each snapshot label data
            label_file_path = self.dummy_label_root_path + save_file_name + "_dummy_fd_ld_labels.csv"

            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                    file_label.close()

                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets(self, file, normalization=False):
        print("INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS...")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_labels.csv".format(self.label_root_path, save_file_name)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")
        selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
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

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
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

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

            # Storing the new snapshot data after processing
            selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)

            appended_edgelist_df = appended_edgelist_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)

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

            window_start_date = window_start_date + dt.timedelta(days=1)


if __name__ == '__main__':
    print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
