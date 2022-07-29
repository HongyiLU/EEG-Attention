import scipy.io as sio
import numpy as np
import pickle
from sklearn import datasets
import torch
from torch.utils.data import Dataset, random_split, TensorDataset
import einops
from einops import rearrange
# class MyDataset(Dataset):

#     def __init__(self, x, y, seq_len):
#         self.data = x
#         self.labels = y
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         if idx + self.seq_len > self.__len__():
#             item = []
#             item[:self.__len__()-idx] = self.data[idx:]
#             return item, self.labels[idx]
#         else:
#             return self.data[idx:idx+self.seq_len], self.labels[idx+self.seq_len]

def create_dataset():

    sequence_length = 16
    fs = 128 
    n_subjects = 5
    mkpt1 = int(fs*10*60)
    mkpt2 = int(fs*20*60)
    mkpt3 = int(fs*30*60)
    subject_map = {}
    for s in range(1, n_subjects+1):
        a =  int(7*(s-1)) + 3
        if s!=5:
            b = a + 5
        else:
            b = a + 4
        subject_map[s] = [i for i in range(a, b)]
    print(subject_map)

    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    useful_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    use_channel_inds = []
    for c in useful_channels:
        if c in channels:
            use_channel_inds.append(channels.index(c))

    inp_dir = 'EEG Data' 

    state_num = {'focussed': 0, 'unfocussed': 1,'drowsed': 2}


    for s in range(1, n_subjects+1):
        data = {}

        data['channels'] = useful_channels
        data['fs'] = fs
        for i, t in enumerate(subject_map[s]):
            trial = {}
            trial_data = sio.loadmat(inp_dir + f'/eeg_record{t}.mat')
            eeg = trial_data['o']['data'][0][0][:, 3:17]
            eeg = eeg[:, use_channel_inds]
            trial['focussed'] = eeg[:mkpt1]
            trial['unfocussed'] = eeg[mkpt1:mkpt2]
            trial['drowsed'] = eeg[mkpt2:mkpt3]
            data[f'trial_{i+1}'] = trial
        with open(f'subject_{s}.pkl', 'wb') as f: 
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    #load 5 subjects' data
    with open('subject_1.pkl', 'rb') as f: 
        data1 = pickle.load(f)
    with open('subject_2.pkl', 'rb') as f: 
        data2 = pickle.load(f)
    with open('subject_3.pkl', 'rb') as f: 
        data3 = pickle.load(f)
    with open('subject_4.pkl', 'rb') as f: 
        data4 = pickle.load(f)
    with open('subject_5.pkl', 'rb') as f: 
        data5 = pickle.load(f)

    #difine label set and data set     
    x = list()
    y = list()   

    for i in range(5):
        for k in data1[f'trial_{i+1}'].keys():
            for d in data1[f'trial_{i+1}'][k]:
                x.append(d)
            for j in range(int(len(data1[f'trial_{i+1}'][k])/sequence_length)):
                y.append(state_num[k])
    
    
    for i in range(5):
        for k in data2[f'trial_{i+1}'].keys():
            for d in data2[f'trial_{i+1}'][k]:
                x.append(d)
            for j in range(int(len(data2[f'trial_{i+1}'][k])/sequence_length)):
                y.append(state_num[k])

    for i in range(5):
        for k in data3[f'trial_{i+1}'].keys():
            for d in data3[f'trial_{i+1}'][k]:
                x.append(d)
            for j in range(int(len(data3[f'trial_{i+1}'][k])/sequence_length)):
                y.append(state_num[k])
    
    #Wrong data
    # for i in range(5):
    #     for k in data4[f'trial_{i+1}'].keys():
    #         for d in data4[f'trial_{i+1}'][k]:
    #             x.append(d)
    #         for j in range(int(len(data4[f'trial_{i+1}'][k])/sequence_length)):
    #             y.append(state_num[k])
    
    for i in range(4):
        for k in data5[f'trial_{i+1}'].keys():
            for d in data5[f'trial_{i+1}'][k]:
                x.append(d)
            for j in range(int(len(data5[f'trial_{i+1}'][k])/sequence_length)):
                y.append(state_num[k])
    
    x_set = np.array(x)
    print(x_set.shape)
    x_set = rearrange(x_set, '(b s) c -> b s c', s = sequence_length)
    x_set = np.expand_dims(x_set, 1).astype(np.float32)
    # x_set = x_set.reshape(-1,sequence_length,len(useful_channels))
    y_set = np.array(y).astype(np.int64)
    
    
    # standardization
    # mean = np.mean(x_set)
    # std = np.std(x_set)
    # x_set = (x_set-mean)/std
    # shuffle_index = np.random.permutation(len(x_set))
    # print(shuffle_index.shape)
    # x_set = x_set[shuffle_index]
    # y_set = y_set[shuffle_index]
    # print(y_set)
    print(x_set.shape)
    print(y_set.shape)
    
    dataset = TensorDataset(torch.from_numpy(x_set), torch.from_numpy(y_set))
    # dataset = MyDataset(x_set, y_set, 128)
    train_size = int(0.7*len(x_set))
    val_size = int(0.2*len(x_set))
    test_size = len(x_set) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size,val_size,test_size], generator=torch.Generator().manual_seed(0))
    print(len(test_data))
    return train_data, val_data, test_data


if __name__ == '__main__':
    create_dataset()