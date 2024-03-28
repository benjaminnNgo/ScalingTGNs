import os
from script.utils.data_util import loader, prepare_dir
from BaselineProcess import creatBaselineDatasets


tokens_dir = "../data/input/tokens" #Change the path to where you save all raw token networks( download from gg drive)

for filename in os.listdir(tokens_dir):
    dataset = filename.split(".")[0]
    creatBaselineDatasets(filename)
    loader(dataset=dataset, neg_sample="rnd")
