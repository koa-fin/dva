# -*-Encoding: utf-8 -*-
import argparse
import torch
import numpy as np
import random
from exp.exp_model import Exp_Model
import os
import pandas as pd

fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# Load data
parser.add_argument('--root_path', type=str, default='./data/2016', help='root path of the data files')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--sequence_length', type=int, default=10, help='length of input sequence')
parser.add_argument('--prediction_length', type=int, default=None, help='prediction sequence length')
parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
parser.add_argument('--input_dim', type=int, default=6, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=128, help='encoder dimension')
parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

# Diffusion process
parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
parser.add_argument('--beta_end', type=float, default=1.0, help='end of the beta')
parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

# Bidirectional VAE
parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')
parser.add_argument('--mult', type=float, default=1, help='mult of channels')
parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

# Training settings
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--itr', type=int, default=5, help='experiment times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
parser.add_argument('--zeta', type=float, default=0.5, help='trade off parameter zeta')
parser.add_argument('--eta', type=float, default=1.0, help='trade off parameter eta')

# Device
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.prediction_length is None:
    args.prediction_length = args.sequence_length

print('Args in experiment:')
print(args)

Exp = Exp_Model
results = pd.DataFrame(columns=['Ticker', 'MSE', 'StdDev'])
train_setting = 'tp{}_sl{}'.format(args.root_path.split(os.sep)[-1], args.sequence_length)

for idx, file in enumerate(os.listdir(args.root_path)): # Iterate through all tickers
    print('\n\nRunning on file {} ({}/{})...'.format(file, idx+1, len(os.listdir(args.root_path))))
    args.data_path = file
    ticker = os.path.splitext(file)[0]
    all_mse = []

    for ii in range(0, args.itr):
        setting = args.data_path + '_' + train_setting
        exp = Exp(args)  # single experiment
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        mse = exp.test(setting)
        all_mse.append(mse)
        torch.cuda.empty_cache()

    results = results.append({'Ticker': ticker, 'MSE': np.mean(np.array(all_mse)), 'StdDev': np.std(np.array(all_mse))}, ignore_index=True)

folder_path = './results/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
results.to_csv(folder_path + train_setting + '.csv', index=False)
print(results)
