import prepare_data as data
import model as m
import visualize
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler




train_loader = data.train_loader
test_loader = data.test_loader


input_dim = 672
input_index = 6
d_head = input_dim * input_index
num_encoder_layer = 12
num_decoder_layer = 12
nhead = 8
num_epochs = 100
learning_rate = 0.0001

model = m.Transformer(d_head, nhead, num_encoder_layer, num_decoder_layer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model.train()
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Đánh giá mô hình trên tập kiểm tra
model.eval()
with torch.no_grad():
    test_outputs = model(data.test_x)
test_loss = criterion(test_outputs.squeeze(), data.test_y)
print(f'Test Loss: {test_loss.item():.4f}')

scaler = MinMaxScaler(feature_range=(0, 1))
predictions = scaler.inverse_transform(test_outputs.numpy())

visualize.visualizeData(data.test_y, predictions)