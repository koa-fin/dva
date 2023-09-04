# -*-Encoding: utf-8 -*-
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


class StandardScaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='AAPL.csv'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len( df_raw)
        num_train = int(length*0.7)
        num_test = int(length*0.2)
        num_vali = int(length*0.1)

        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = pd.DatetimeIndex(df_raw[border1:border2]['date'])
        data_stamp = time_features(df_stamp)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end, -1:]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
