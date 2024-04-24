import statistics as stat
import pandas as pd

target_dataset = "unnamedtoken216360xfca59cd816ab1ead66534d82bc21e7515ce441cf"
target_lr = 0.00015
test_auc = []
test_ap = []

results_df = pd.read_csv('../../data/output/i4/results.csv')
for index, row in results_df.iterrows():
    if row['dataset'] == target_dataset and row['lr'] == target_lr:
        test_auc.append(row['test_auc'])
        test_ap.append(row['test_ap'])


print("Result AUCs:",test_auc)
print("Result APs:",test_ap)

print("Test AUC :{},{}".format(stat.mean(test_auc),stat.stdev(test_auc)))
print("Test AP :{},{}".format(stat.mean(test_ap),stat.stdev(test_ap)))