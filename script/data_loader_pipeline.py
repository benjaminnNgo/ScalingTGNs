import os
from script.utils.data_util import loader, prepare_dir
from BaselineProcess import creatBaselineDatasets


tokens_dir = "../data/input/tokens" #Change the path to where you save all raw token networks( download from gg drive)

# for filename in os.listdir(tokens_dir):
#
#     dataset = filename.split(".")[0].replace("_","")
#     creatBaselineDatasets(filename)
#     # loader(dataset=dataset, neg_sample="rnd")

creatBaselineDatasets("unnamed_token_21635_0xe53ec727dbdeb9e2d5456c3be40cff031ab40a55.csv")
