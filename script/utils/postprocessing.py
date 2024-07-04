import pandas as pd
import numpy as np
result_folder = "../data/output/test_result"

def average_result(result_list):
    data_aucs = {}
    data_aps = {}
    for result in result_list:
        result_path = '{}/{}_results.csv'.format(result_folder, result)
        results = pd.read_csv(result_path, index_col=False, header=0)
        print(results)
        for idx, row in results.iterrows():
            if row["mode"] == "Test":
                if row["dataset"] not in data_aucs:
                    data_aucs[row["dataset"]] = [row["test_auc"]]
                    data_aps[row["dataset"]] = [row["test_ap"]]
                else:
                    data_aucs[row["dataset"]].append(row["test_auc"])
                    data_aps[row["dataset"]].append(row["test_ap"])
        print(data_aucs)
        print(data_aps)

average_result(["HTGN_seed_800_1_2024-05-02-22:41:19"])