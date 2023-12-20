import tgx
from utils import * 
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser('*** discretizing time steps from datasets ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('-t', '--time', type=str, help='time granularity', default='daily')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 


"""
1. load a dataset
2. load into a graph
3. discretize the graph
4. save the graph back to a csv
"""

args, _ = get_args()


#! load the datasets
# dataset = tgx.builtin.uci()

data_name = args.data #"tgbl-coin" #"tgbl-review" #"tgbl-wiki"
dataset = tgx.tgb_data(data_name)
# dataset = tgx.tgb_data("tgbl-review")
# dataset = tgx.tgb_data("tgbl-coin") 


ctdg = tgx.Graph(dataset)
# ctdg.save2csv("ctdg")

time_scale = args.time #"minutely"  #"monthly" #"weekly" #"daily"  #"hourly" 
dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
print ("discretize to ", time_scale)
print ("there is time gap, ", dtdg.check_time_gap())
list2csv(ts_list, data_name + "_ts" + "_" + time_scale + ".csv")

# download from https://object-arbutus.cloud.computecanada.ca/tgb/dtdg_ts.zip


#* tgbl-coin download from https://object-arbutus.cloud.computecanada.ca/tgb/dtdg_coin_ts.zip
