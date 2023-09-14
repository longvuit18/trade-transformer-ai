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
scaled_data = scaler.fit_transform(data[['close', 'ema2', 'ema9', 'ema25', 'ema97', 'volume']].values)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data = scaled_data[:800]
test_data = scaled_data[800:]

# Hàm tạo batch từ dữ liệu chuỗi thời gian
def create_sequences(data, seq_length, key):
    
    close = []
    ema2 = []
    ema9 = []
    ema25 = []
    ema97 = []
    volume = []
    for i in range(len(data) - seq_length):
        print()
        sequence = data[i:i + seq_length]
        for k in range(len(sequence)):
            close = 
        target = data[i + seq_length][1]
        sequences.append((sequence, target))
    return sequences

# Định nghĩa kiến trúc mô hình Transformer
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x.transpose(0, 1))  # Chuyển vị x trước khi đưa vào encoder
        x = self.decoder(x[-1])  # Dự đoán trường 'close' từ kết quả cuối cùng
        return x

# Cài đặt các siêu tham số
input_dim = 1  # Độ dài của chuỗi đầu vào (giá trị Bitcoin)
d_model = 128  # Kích thước của vector biểu diễn cho mỗi phần tử trong chuỗi
nhead = 4  # Số lượng đầu vào đầu ra độc lập được sử dụng trong self-attention
num_layers = 3  # Số lượng lớp mã hóa tổng cộng

# Khởi tạo mô hình và bộ tối ưu
model = Transformer(input_dim, d_model, nhead, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Chuyển đổi dữ liệu huấn luyện và tạo batch
train_sequences = create_sequences(train_data, seq_length=32)
train_inputs = torch.tensor([seq for seq, _ in train_sequences], dtype=torch.float32)
train_targets = torch.tensor([target for _, target in train_sequences], dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(train_inputs.transpose(1, 2), train_targets)  # Chuyển vị train_inputs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# Huấn luyện mô hình
# num_epochs = 100
# model.train()
# for epoch in range(num_epochs):
#     for batch_inputs, batch_targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_inputs.unsqueeze(2))
#         loss = criterion(outputs.squeeze(), batch_targets)
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# # Đánh giá mô hình trên tập kiểm tra
# model.eval()
# test_sequences = create_sequences(test_data, seq_length=10)
# test_inputs = torch.tensor([seq for seq, _ in test_sequences], dtype=torch.float32)
# test_targets = torch.tensor([target for _, target in test_sequences], dtype=torch.float32)
# with torch.no_grad():
#     test_outputs = model(test_inputs.unsqueeze(2))
# test_loss = criterion(test_outputs.squeeze(), test_targets)
# print(f'Test Loss: {test_loss.item():.4f}')

# # Giải mã dữ liệu dự đoán
# predictions = scaler.inverse_transform(test_outputs.numpy())

# # Vẽ đồ thị giá Bitcoin thực tế và dự đoán
# import matplotlib.pyplot as plt

# plt.plot(data['Close'].values[800 + 10:], label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
