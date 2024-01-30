import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, future_unknown_days = 0, shift_data_y = 'Yes', anio = 'anio3'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.anio = anio
        
        #New variables
        self.future_unknown_days = future_unknown_days
        self.future_unknown_hours = future_unknown_days * 24
        #Boolean variables regarding unknown days
        self.shift_data_y = shift_data_y  == "Yes"
        self.split_in_get_items = not self.shift_data_y
        self.use_x_data_for_labels = self.shift_data_y
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        
        unknown_hours = self.future_unknown_hours
        
        if self.data_path.startswith("greek_energy"):
            if self.anio == 'no_anio':
                # 2010 - 2016 training : Rows[2-54817] -> 54816 rows != 6.2 years - 2284 * 24
                # 2017 validation  : Rows[54818-63577] -> 8760 rows  = 1 year - 365 * 24
                # 2018 test        : Rows[63578-72337] -> 8760 rows  = 1 year - 365 * 24
                # Total rows: 72336
                train_rows = 2284*24
                val_rows = 365*24
                test_rows = 365*24
                
                border1s = [0, train_rows - self.seq_len - unknown_hours , train_rows + val_rows - self.seq_len - unknown_hours]
                border2s = [train_rows - unknown_hours , train_rows + val_rows - unknown_hours , train_rows + val_rows + test_rows - unknown_hours]
            
            else:
                # 2010 - 2016 training : Rows[2-54817] -> difference
                # 2017 validation  : Rows[54818-63577] -> 8760 rows  = 1 year - 365 * 24
                # 2018 test        : Rows[63578-72337] -> 8760 rows  = 1 year - 365 * 24
                # Total rows: 72336
                val_rows = 365*24
                test_rows = 365*24
                train_rows = int((len(df_raw) / 24 - 730)*24)
                
                border1s = [0, train_rows - self.seq_len - unknown_hours , train_rows + val_rows - self.seq_len - unknown_hours]
                border2s = [train_rows - unknown_hours , train_rows + val_rows - unknown_hours , train_rows + val_rows + test_rows - unknown_hours]
        elif self.data_path.startswith("ETTh"):
            # ETTh datasets have 17421 values 
            # ETTh datasets have some less values about 16581 - 16749 depending on the case
            # A split of 12 - 4 - 4 months gives a total of 14400 values which is good for all the cases
            train_rows = 12*30*24
            val_rows = 4*30*24
            test_rows = 4*30*24
                
            border1s = [0, train_rows - self.seq_len - unknown_hours , train_rows + val_rows - self.seq_len - unknown_hours]
            border2s = [train_rows - unknown_hours , train_rows + val_rows - unknown_hours , train_rows + val_rows + test_rows - unknown_hours]   
              
        border1_x = border1s[self.set_type]
        border2_x = border2s[self.set_type]
        if self.shift_data_y:
            border1s_y = [x + unknown_hours  for x in border1s]
            border2s_y = [x + unknown_hours  for x in border2s]
            border1_y = border1s_y[self.set_type]
            border2_y = border2s_y[self.set_type]
        else:
            border1_y = border1s[self.set_type]
            border2_y = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1_x:border2_x]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
                    
        self.data_x = data[border1_x:border2_x]
        if self.inverse:
            self.data_y = df_data.values[border1_y:border2_y]
        else:
            self.data_y = data[border1_y:border2_y]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        seq_x = self.data_x[s_begin:s_end]  # 0 - 24
        seq_x_mark = self.data_stamp[s_begin:s_end]
        
        r_begin1 = s_end - self.label_len 
        r_end1 = r_begin1 + self.label_len
        
        if self.split_in_get_items:
            r_begin2 = r_end1 + self.future_unknown_hours
        else:
            r_begin2 = r_end1
        
        r_end2 = r_begin2 + self.pred_len
        
        if self.use_x_data_for_labels:    
            seq_y = np.vstack((self.data_x[r_begin1:r_end1],self.data_y[r_begin2:r_end2]))
            seq_y_mark = np.vstack((self.data_stamp[r_begin1:r_end1],self.data_stamp[r_begin2:r_end2]))
        else:
            seq_y = np.vstack((self.data_y[r_begin1:r_end1],self.data_y[r_begin2:r_end2]))
            seq_y_mark = np.vstack((self.data_stamp[r_begin1:r_end1],self.data_stamp[r_begin2:r_end2]))
            
        print_and_save_index_0 = False
            
        if index == 0 and print_and_save_index_0:
            inv_seq_x = self.scaler.inverse_transform(seq_x)
            inv_seq_y = self.scaler.inverse_transform(seq_y)
                
            print("inv_seq_x.shape: ", inv_seq_x.shape)
            print("inv_seq_x[:,-1]: ", inv_seq_x[:,-1])
            print("inv_seq_y.shape: ", inv_seq_y.shape)
            print("inv_seq_y[:,-1]: ", inv_seq_y[:,-1])
            print(f"(index, s_begin, s_end, r_begin1, r_end1, r_begin2, rend2) = ({index}, {s_begin}, {s_end}, {r_begin1}, {r_end1}, {r_begin2}, {r_end2})")
            
            if False:
                inv_seq_x_file = '{}ukn_{}shifty_inv_seq_x[a,-1].npy'.format(self.future_unknown_days, self.shift_data_y)
                inv_seq_y_file = '{}ukn_{}shifty_inv_seq_y[a,-1].npy'.format(self.future_unknown_days, self.shift_data_y)
                np.save(inv_seq_x_file, inv_seq_x[:,-1])
                np.save(inv_seq_y_file, inv_seq_y[:,-1])
            

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len - self.future_unknown_hours + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
