import os
import random

# random.seed(401)
#
#
# path = '../data/input/cached'
# datasets = []
# for dataset in os.listdir(path):
#     datasets.append(dataset)
#
# random.shuffle(datasets)
#
# print(len(datasets))
# # i = 0
# # while 2**i <= len(datasets):
# #     package_size = 2**i
# #     subdatasets = datasets[:2**i]
# #     with open('../data/datasets_package_{}.txt'.format(package_size),'w') as file:
# #         for dataset in subdatasets:
# #             file.write(str(dataset) + "\n")
# #     i+= 1
#
# subdatasets = datasets[-20:]
# with open('../data/datasets_testing_package.txt','w') as file:
#         for dataset in subdatasets:
#             file.write(str(dataset) + "\n")


import pandas as pd

# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Age': [25, 30, 35, 40, 45]
# }
#
# df = pd.DataFrame(data)
# for i in range(len(df)):
#     df.loc[i, 'Name'] = "Bao"
#
# print(df)

files = ['unnamed_token_1898_0x00a8b738e453ffd858a7edf03bccfe20412f0eb0.csv','unnamed_token_21624_0x83e6f1e41cdd28eaceb20cb649155049fac3d5aa.csv']

first_token_df = pd.read_csv("E:/token/"+files[0])
second_token_df = pd.read_csv("E:/token/"+files[1])

node_address_1 = set()
node_address_1.update(first_token_df['from'].tolist())
node_address_1.update(first_token_df['to'].tolist())

print(1,len(node_address_1))
node_address_2 = set()
node_address_2.update(second_token_df['from'].tolist())
node_address_2.update(second_token_df['to'].tolist())

print(2,len(node_address_2))

intersection = node_address_2 & node_address_1


print(len(intersection))