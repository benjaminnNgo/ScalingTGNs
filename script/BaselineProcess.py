import os
import shutil
import datetime as dt
import pandas as pd

root_path = "../data/Tokens/"
timeseries_file_path = "../data/Tokens/"
timeseries_file_path = '/network/scratch/r/razieh.shirzadkhani/fm_data/selected/'

def creatBaselineDatasets(file, normalization=False):
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    minValidDuration = 20  # Day
    indx = 0

    if os.path.exists("../data/Baseline/Labels/" + file):
        os.remove("../data/Baseline/Labels/" + file)
    selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=',')
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
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
    if normalization:
        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    # Graph Generation Process and Labeling

    base_progress = (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nCompleted Process  {} % ".format(

            (1 - ((data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)) / base_progress) * 100))
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
        csv_file_path = "../data/Baseline/" + file
        if not os.path.exists(csv_file_path):
            selectedNetworkInGraphDataWindow.to_csv(csv_file_path, mode='a', index=False)
        else:
           
            selectedNetworkInGraphDataWindow.to_csv(csv_file_path, mode='a', index=False, header=False)
        # ------------------------------------------------
        # Storing each snapshot label data
        label_file_path = "../data/Baseline/Labels/" + file
        # Open a file in append mode and write a line to it
        with open(label_file_path, 'a') as file_label:
            file_label.write(str(label)+"\n")
            file_label.close()
        # --------------------------------------------------

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Process completed! 100%")



# ["unnamed_token_15_0x0000000000095413afc295d19edeb1ad7b71c952.csv",
# "unnamed_token_21662_0x429881672b9ae42b8eba0e26cd9c73711b891ca5.csv",
# "unnamed_token_21658_0x5f98805a4e8be255a32880fdec7f6728c6568ba0.csv",
# "unnamed_token_21655_0xbcca60bb61934080951369a648fb03df4f96263c.csv",
# "unnamed_token_21654_0x09a3ecafa817268f77be1283176b946c4ff2e608.csv",
# "unnamed_token_21639_0x1ceb5cb57c4d4e2b2433641b95dd330a33185a44.csv",
# "unnamed_token_21624_0x83e6f1e41cdd28eaceb20cb649155049fac3d5aa.csv",
# "unnamed_token_1898_0x00a8b738e453ffd858a7edf03bccfe20412f0eb0.csv",
# "unnamed_token_21630_0xcc4304a31d09258b0029ea7fe63d032f52e44efe.csv",
# "unnamed_token_21636_0xfca59cd816ab1ead66534d82bc21e7515ce441cf.csv"]