
# Vẽ đồ thị giá Bitcoin thực tế và dự đoán
import matplotlib.pyplot as plt


def visualizeData(currentData, predictions):
  plt.plot(currentData, label='Actual')
  plt.plot(predictions, label='Predicted')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
