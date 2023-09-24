import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dữ liệu Bitcoin từ tệp CSV
data = pd.read_csv('normalized-data.csv')

# Chuẩn hóa dữ liệu vào khoảng [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(
    data[['close', 'ema2', 'ema9', 'ema25', 'ema97', 'volume']].values)
# Hàm tạo batch từ dữ liệu chuỗi thời gian


def create_sequences(data, seq_length, output_length=96):
    input = []
    target = []
    for i in range(len(data) - seq_length - output_length):
        sequence = data[i:i + seq_length]
        closeList = []
        ema2List = []
        ema9List = []
        ema25List = []
        ema97List = []
        volumeList = []
        ema2ListOutput = []
        for k in range(len(sequence)):
            closeList.append(sequence[k][0])
            ema2List.append(sequence[k][1])
            ema9List.append(sequence[k][2])
            ema25List.append(sequence[k][3])
            ema97List.append(sequence[k][4])
            volumeList.append(sequence[k][5])
        input_matrix = np.stack(
            [closeList, ema2List, ema9List, ema25List, ema97List, volumeList], axis=1)
        for k in range(i + seq_length, i + seq_length + output_length):
            ema2ListOutput.append(data[k][1])
        input.append(input_matrix)
        target.append(ema2ListOutput)
    return np.array(input), np.array(target)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

trains_in, trains_target = create_sequences(train_data, seq_length=672)
tests_in, test_target = create_sequences(train_data, seq_length=672)

print("chieu cua trains input data: ", trains_in.shape)
print("chieu cua trains target data: ", trains_target.shape)

print("chieu cua test input data: ", trains_in.shape)
print("chieu cua test target data: ", trains_target.shape)
trans_shape = trains_target.shape

train_x = torch.from_numpy(trains_in).float()
train_y = torch.from_numpy(trains_target).float()
test_x = torch.from_numpy(tests_in).float()
test_y = torch.from_numpy(test_target).float()
# tạo dataloader
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(train_x, train_y)
test_data = TensorDataset(test_x, test_y)

# tạo dataloader
batch_size = 96
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
# kiểm tra kích thước của tập dữ liệu
# print(len(train_loader)) #1428
# print(len(test_loader)) #348





