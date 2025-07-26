import torch
import torch.nn as nn
import torch.nn.functional as f


class ProteinFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers=2, dropout_rate=0.5, kernel_size=5):
        super(ProteinFeatureExtractor, self).__init__()
        self.input_size = input_size  # 确保这与one_hot_vector_length相匹配
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=kernel_size, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)  

        # LSTM层
        # LSTM的input_size应为卷积层的输出通道数乘以双向因子（如果是双向LSTM）
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=hidden_size_1,
                             num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size_1 * 2, hidden_size=hidden_size_2,
                             num_layers=num_layers, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        # x是[seq_length, esm]
        # 因为是1D卷积，所以先调整为[1, seq_length, esm]
        x = data.x.unsqueeze(0)  # [1, seq_length, esm]
        h = x
        h = h.permute(0, 2, 1)
        # 卷积层
        h = f.relu(self.conv1(h))
        h = f.relu(self.conv2(h))
        h = self.pool(h)  # 池化后形状大致为 [batch_size, channels, seq_length // 4]
        h = h.permute(0, 2, 1)  # [batch_size, seq_length // 5, channels]

        # LSTM层
        h, _ = self.lstm1(h)
        h = self.dropout(h)
        h, _ = self.lstm2(h)
       
        # 展平
        h = torch.flatten(h)
    
        return h
