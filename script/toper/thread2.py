import pandas as pd

from script.toper.TopER import generate_toper_from_list

if __name__ == '__main__':
    TGS_data = pd.read_csv("../../data/TGS_available_datasets.csv")

    data_list = TGS_data['dataset'].tolist()
    n = len(data_list)
    size = n // 3
    list1 = data_list[:size]
    list2 = data_list[size:2 * size]
    list3 = data_list[2 * size:]
    generate_toper_from_list(list3)