import statistics as stat
import pandas as pd
import numpy as np

# target_dataset = "unnamedtoken216540x09a3ecafa817268f77be1283176b946c4ff2e608"
# # target_lr = 0.00015
# test_auc = []
# test_ap = []
#
# results_df = pd.read_csv('../../data/output/single_model_egcn/results_lab.csv')
# for index, row in results_df.iterrows():
#     if row['dataset'] == target_dataset:
#         test_auc.append(row['test_auc'])
#         test_ap.append(row['test_ap'])
#
#
# print("Result AUCs:",test_auc)
# print("Result APs:",test_ap)
#
# print("Test AUC :{},{}".format(stat.mean(test_auc),stat.stdev(test_auc)))
# print("Test AP :{},{}".format(stat.mean(test_ap),stat.stdev(test_ap)))

def get_mean_std(target_dataset,result_file):
    results_df = pd.read_csv(result_file)
    test_auc = []
    test_ap = []
    for index, row in results_df.iterrows():
        if row['dataset'] == target_dataset:
            test_auc.append(row['test_auc'])
            test_ap.append(row['test_ap'])

    assert len(test_auc) == 3, "Missing results"
    assert len(test_ap) == 3, "Missing results"
    return stat.mean(test_auc),stat.stdev(test_auc),stat.mean(test_ap),stat.stdev(test_ap)

def get_dummy_results(dataset):
    results_df = pd.read_csv('../../data/output/baselinemodel.csv')
    data_row = results_df[results_df['dataset'] == dataset]
    return float(data_row['auc']),float(data_row['ap'])

def get_rank_arr(values):
    # sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k], reverse=True)

    sorted_indices = np.argsort(values)[::-1]

    # Create a rank array of the same length
    rank = [0] * len(values)

    # Assign ranks to the elements based on the sorted order
    for i, idx in enumerate(sorted_indices):
        rank[idx] = i + 1

    return rank

if __name__ == '__main__':
    target_dataset = [
        "unnamedtoken216540x09a3ecafa817268f77be1283176b946c4ff2e608",
        "unnamedtoken223250xf2ec4a773ef90c58d98ea734c0ebdb538519b988",
        "unnamedtoken222800xa49d7499271ae71cd8ab9ac515e6694c755d400c",
        "unnamedtoken223030x4ad434b8cdc3aa5ac97932d6bd18b5d313ab0f6f",
        "unnamedtoken220850x9fa69536d1cda4a04cfb50688294de75b505a9ae",
        "unnamedtoken220220xade00c28244d5ce17d72e40330b1c318cd12b7c3",
        "unnamedtoken223090xc4ee0aa2d993ca7c9263ecfa26c6f7e13009d2b6",
        "unnamedtoken221090x5de8ab7e27f6e7a1fff3e5b337584aa43961beef",
        "unnamedtoken220240x235c8ee913d93c68d2902a8e0b5a643755705726",
        "unnamedtoken221150xa2cd3d43c775978a96bdbf12d733d5a1ed94fb18",
        "unnamedtoken218340xaa6e8127831c9de45ae56bb1b0d4d4da6e5665bd",
        "unnamedtoken220960x4da27a545c0c5b758a6ba100e3a049001de870f5",
        "unnamedtoken217780x7dd9c5cba05e151c895fde1cf355c9a1d5da6429",
        "unnamedtoken220250xa71d0588eaf47f12b13cf8ec750430d21df04974",
        "unnamedtoken218270x5026f006b85729a8b14553fae6af249ad16c9aab",
        "unnamedtoken221900x49642110b712c1fd7261bc074105e9e44676c68f",
        "unnamedtoken216900x9e32b13ce7f2e80a01932b42553652e053d6ed8e",
        "unnamedtoken218450x221657776846890989a759ba2973e427dff5c9bb",
        "TRAC0xaa7a9ca87d3694b5755f213b5d04094b8d0f0a6f",
        "unnamedtoken220280xcf3c8be2e2c42331da80ef210e9b1b307c03d36a",
    ]
    model = "htgn"

    single_result_file_path = '../../data/output/single_model_{}_test/results.csv'.format(model)
    foundation_model_result_path = '../../data/output/foundation_model_{}/results_auc.csv'.format(model)
    foundation_models_all_results = pd.read_csv(foundation_model_result_path)
    #
    # models = ['dummy','single','foundation-1','foundation-2','foundation-4','foundation-8','foundation-16','foundation-32','foundation-64']
    # row_labels = ['best_count','avg_rank']
    # rank_dict = {
    #     'dummy':[],
    #     'single':[],
    #     'foundation-1':[],
    #     'foundation-2':[],
    #     'foundation-4':[],
    #     'foundation-8':[],
    #     'foundation-16':[],
    #     'foundation-32':[],
    #     'foundation-64':[]
    # }
    # rowlist =[]
    # all_dummy_auc = []
    # all_dummy_ap = []
    # all_single_auc = []
    # all_single_ap = []
    # for dataset in target_dataset:
    #     auc_values = []
    #     foundation_models_results = foundation_models_all_results[foundation_models_all_results['dataset'] == dataset]
    #     dummy_auc,dummy_ap = get_dummy_results(dataset)
    #     all_dummy_auc.append(dummy_auc)
    #     all_dummy_ap.append(dummy_ap)
    #     auc_values.append(dummy_auc)
    #
    #     single_auc,_,single_ap,_ =get_mean_std(dataset,single_result_file_path)
    #     all_single_auc.append(single_auc)
    #     all_single_ap.append(single_ap)
    #     auc_values.append(single_auc)
    #
    #     log_2 = (1, 2, 4, 8, 16, 32, 64)
    #     for index in log_2:
    #         auc_values.append(float(foundation_models_results['foundation-{}'.format(index)]))
    #     print(auc_values)
    #     rank_arr = get_rank_arr(auc_values)
    #     rank_dict['dummy'].append(rank_arr[0])
    #     rank_dict['single'].append(rank_arr[1])
    #     counter = 2
    #     for index in log_2:
    #         rank_dict['foundation-{}'.format(index)].append(rank_arr[counter])
    #         counter+=1
    #
    # row_one_count = []
    # avg_row = []
    # for model in rank_dict:
    #    row_one_count.append(rank_dict[model].count(1))
    #    avg_row.append(np.mean(rank_dict[model]))
    #
    # rowlist.append(row_one_count)
    # rowlist.append(avg_row)
    #
    # df = pd.DataFrame(rowlist,columns=models,index=row_labels)
    # print(df)
    # df.to_csv("htgn_results.csv")
    #
    # print("Dummy:",stat.mean(all_dummy_auc),stat.mean(all_dummy_ap))
    # print("Single:", stat.mean(all_single_auc), stat.mean(all_single_ap))


#check gclstm
    # all_auc = []
    # for dataset in target_dataset:
    #     single_auc,_,single_ap,_ =get_mean_std(dataset,single_result_file_path)
    #
    #     # if(dataset != "unnamedtoken220240x235c8ee913d93c68d2902a8e0b5a643755705726" and dataset != "unnamedtoken220960x4da27a545c0c5b758a6ba100e3a049001de870f5"):
    #     all_auc.append(single_auc)
    #     print(dataset," AUC:",single_auc)
    #
    # print("Mean:",stat.mean(all_auc), "STD:",stat.stdev(all_auc))
    columns = ["dataset","HTGN-AUC","GCLSTM-AUC","dummy-AUC","HTGN-AP","GCLSTM-AP","dummy-AP"]
    token_mapper = pd.read_csv("../../data/TGS_available_datasets.csv")

    rowlist = []
    for dataset in target_dataset:
        single_auc_gclstm, std_gclstm_auc, single_ap_gclstm,std_gclstm_ap  = get_mean_std(dataset, '../../data/output/single_model_gclstm_test/results.csv')
        single_auc_htgn, std_htgn_auc, single_ap_htgn, std_htgn_ap = get_mean_std(dataset,
                                                                 '../../data/output/single_model_htgn_test/results.csv')

        dummy_auc,dummy_ap = get_dummy_results(dataset)

        token_name = token_mapper[token_mapper["dataset"] == dataset]["token_name"].values[0]
        rowlist.append([dataset,single_auc_htgn,single_auc_gclstm,dummy_auc,single_ap_htgn,single_ap_gclstm,dummy_ap])
        print(dataset,token_name)

    df = pd.DataFrame(rowlist, columns=columns)
    print(df)









