import argparse
import torch
import os

parser = argparse.ArgumentParser(description='HTGN')
# 1.dataset
parser.add_argument('--dataset', type=str, default='enron10', help='datasets')
parser.add_argument('--data_pt_path', type=str, default='', help='need to be modified')
parser.add_argument('--num_nodes', type=int, default=33190, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=5, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=16, help='dim of hidden embedding')  # 32-64
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')
parser.add_argument('--neg_sample', type=str, default='rnd', help='negative sampling strategy')
parser.add_argument("--wandb", action="store_true", default=False, help="now using wandb")
parser.add_argument('--project', type=str, default='Foundation_TGCN_train', help='wandb project name')
parser.add_argument('--entity', type=str, default='kia-team', help='wandb entity name')
parser.add_argument('--results_file', type=str, default='results.csv',
                    help='Name of file to store evaluation of all models')
parser.add_argument('--data_pack_train', type=str, default='pack_2', help='data package used for training')
parser.add_argument('--data_pack_test', type=str, default='pack_test', help='data package used for testing')
parser.add_argument('--data_pack_idx_start', type=int, default=0,
                    help='start index of data package in the list of data packages')
parser.add_argument('--data_pack_idx_end', type=int, default=1,
                    help='end index of data package in the list of data packages')
parser.add_argument('--use_dos', action="store_true", default=True, help="whether to use DOS as graph-level features")
parser.add_argument('--normalize_dos', action="store_true", default=True, help="whether to normilize DOS vector")
parser.add_argument('--concat_att_dos_to_original', action="store_true", default=True,
                    help="Use both riginal snapshot DOS and the attention based DOS together")
parser.add_argument('--set_green_readout_TGCN', action="store_true", default=True,
                    help="Just use the current snapshot node size history for pooling at the end of TGCN")
parser.add_argument('--set_green_hidden_layer_TGCN', action="store_true", default=True,
                    help="Just use the current snapshot node size history for hidden layer of TGCN")
parser.add_argument('--dos_memory_size', type=int, default=5,
                    help='number of epochs to train. help="Just use the current snapshot node size history for hidden layer of TGCN')

parser.add_argument('--loss_comb_method', type=str, default='simple_sum', help='Loss Combination Parameter')
parser.add_argument('--pack',type=int, default=4, help='Pack shortcut')
# 2.experiments
parser.add_argument('--max_epoch', type=int, default=500, help='number of epochs to train.')
parser.add_argument('--testlength', type=int, default=3, help='length for test, default:3')
parser.add_argument('--evalLength', type=int, default=3, help='length for eval, default:3')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--repeat', type=int, default=1, help='running times')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
parser.add_argument('--use_htc', type=int, default=1, help='use htc or not, default: 1')
parser.add_argument('--use_hta', type=int, default=1, help='use hta or not, default: 1')
parser.add_argument('--debug_content', type=str, default='', help='debug_mode content')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--node_feats', action="store_true", default=True, help='node feature')
parser.add_argument('--save_embeddings', type=int, default=0, help='save or not, default:0')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')
parser.add_argument('--trainratio', type=float, default=0, help='for scaling on train ratio')
parser.add_argument('--fixed_train_length', type=int, default=20, help='for scaling on train snapshots')
parser.add_argument('--test_ratio', type=int, default=0.15, help='ratio for testing, default:0.15')
parser.add_argument('--val_ratio', type=int, default=0.15, help='ratio for evaluation, default:0.15')
parser.add_argument('--curr_time', type=str, default='0', help='local time for run to be used for saved models')
# For testing model only
parser.add_argument('--test_dataset', type=str, default=None, help='define dataset for testing')
parser.add_argument('--test_snapshot', type=str, default=5, help='define test snapshot for testing')
parser.add_argument('--single_task', type=str, default='edge_gs', help='single task for the model training')
# 3.models
parser.add_argument('--model', type=str, default='Hydra_Multi_decoder_att_on_dos', help='models name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='Hyperbolic models')
parser.add_argument('--use_gru', type=bool, default=True, help='use gru or not')
parser.add_argument('--use_hyperdecoder', type=bool, default=True, help='use hyperbolic decoder or not')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--nb_window', type=int, default=5, help='the length of window')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=int, default=0,
                    help='using trainable feat or one-hot feat, default: none-trainable feat')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=1, help='attention heads.')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--curvature', type=float, default=1.0, help='curvature value')
parser.add_argument('--fixed_curvature', type=int, default=1, help='fixed (1) curvature or not (0)')
parser.add_argument('--aggregation', type=str, default='deg', help='aggregation method: [deg, att]')
parser.add_argument('--test_bias', type=bool, default=False, help='Test for the bias in model testing')
parser.add_argument('--graph_agg', type=str, default='mean', help='graph aggregation strategy')
parser.add_argument('--use_multi_head', type=bool, default=True,
                    help='Use multi-head for multiple properties prediction')
parser.add_argument('--model_parameter_scaling_factor', type=int, default=1, help='The scaling factor on number of model parameters')



args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('INFO: using gpu:{} to train the models'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('INFO: using cpu to train the models')

print(args.model)
args.output_folder = '../data/output/log/{}/{}/'.format(args.dataset, args.model)
args.result_txt = '../data/output/results/{}_{}_result.txt'.format(args.dataset, args.model)

# open debugging mode
if args.debug_mode == 1:
    print('start debugging mode!')
    folder = '../data/output/ablation_study/{}/'.format(args.debug_content)
    args.result_txt = folder + '{}_{}_result.txt'.format(args.dataset, args.model)
    if not os.path.isdir(folder):
        os.makedirs(folder)
