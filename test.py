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
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data = scaled_data[:800]
test_data = scaled_data[800:]

# Hàm tạo batch từ dữ liệu chuỗi thời gian


def create_sequences(data, seq_length, output_length = 92):
    sequences = []
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
        input_matrix = np.stack([closeList, ema9List, ema25List, ema97List, volumeList], axis=1)
        for k in range(i + seq_length, i + seq_length + output_length):
            ema2ListOutput.append(data[k][1])
        sequences.append((input_matrix, ema2ListOutput))
    return sequences

# Định nghĩa kiến trúc mô hình Transformer
class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(5, 8, 2048), 6)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(5, 8, 2048), 6)
        self.linear = nn.Linear(5, 1)

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = output.permute(1, 0, 2)
        output = self.linear(output)
        return output
# Cài đặt các siêu tham số
batch_size = 64
seq_length = 128
output_length = 92
num_epochs = 100
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình và bộ tối ưu
model = Transformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
# Chuyển đổi dữ liệu huấn luyện và tạo batch
train_sequences = create_sequences(train_data, seq_length=644)
# print(train_sequences[0])
train_inputs = torch.tensor(
    np.array([seq for seq, _ in train_sequences]), dtype=torch.float32)
train_targets = torch.tensor(
    np.array([target for _, target in train_sequences]), dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(
    train_inputs.transpose(1, 2), train_targets)  # Chuyển vị train_inputs
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

#Huấn luyện mô hình
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs.unsqueeze(2))
        loss = criterion(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Đánh giá mô hình trên tập kiểm tra
model.eval()
test_sequences = create_sequences(test_data, seq_length=10)
test_inputs = torch.tensor([seq for seq, _ in test_sequences], dtype=torch.float32)
test_targets = torch.tensor([target for _, target in test_sequences], dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(test_inputs.unsqueeze(2))
test_loss = criterion(test_outputs.squeeze(), test_targets)
print(f'Test Loss: {test_loss.item():.4f}')