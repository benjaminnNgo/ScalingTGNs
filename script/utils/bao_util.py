import json
import pandas as pd

df = pd.read_csv("../../data/output/single_model_train_set/results.csv")
datasets_list = pd.read_csv("../../data/data_package/datasets_package_64.txt").iloc[:,0].values
seeds = set([710,720,800])
missing_dict = {}
count = 0
for data in datasets_list:
    rows = df[df['dataset'] == data]
    if rows.shape[0] < 3:
        ran_seed = set(rows['seed'].tolist())
        missing_seed = seeds.difference()
        missing_dict[data]= list(missing_seed)
        count += len(missing_seed)


print(len(missing_dict))
with open('../../data/missing_tokens.json', 'w') as json_file:
    json.dump(missing_dict, json_file)