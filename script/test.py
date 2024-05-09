from script.bao_util import compute_reocurrence_surprise_from_datasets, plot_hist
import pandas as pd
TGS_available_dataset = pd.read_csv('../data/data_package/datasets_package_64.txt').iloc[:,0].tolist()
reoccurrence_list, surprise_list = compute_reocurrence_surprise_from_datasets(TGS_available_dataset)

plot_hist(reoccurrence_list)
plot_hist(surprise_list)