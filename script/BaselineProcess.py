import os
import shutil
import datetime as dt
import pandas as pd

root_path = "../data/input/temp"
timeseries_file_path = "../data/input/temp/"

def creatBaselineDatasets(file, normalization=False):
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0

    selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=' ', names=["from", "to", "date", "value"])
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
        csv_file_path = "../data/input/raw/edgelists/" + file
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
        else:
            existing_df = pd.DataFrame()

        appended_df = existing_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
        appended_df.to_csv(csv_file_path, index=False)
        # ------------------------------------------------
        # Storing each snapshot label data
        label_csv_file_path = "../data/input/raw/labels/"  + file
        if os.path.exists(label_csv_file_path):
            existing_label_df = pd.read_csv(label_csv_file_path)
        else:
            existing_label_df = pd.DataFrame()

        appended_label_df = existing_label_df.append(label, ignore_index=True)
        appended_label_df.to_csv(label_csv_file_path, index=False)
        # --------------------------------------------------

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed!")
