import torch
import torch.nn as nn
from config.config import Config
from model.modules.feature_LSTM import ProteinFeatureExtractor
from model.modules.feature_GCN import GraphConvNet
from model.modules.coformer import CoFormer


class ProAffinity(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, in_channels, hidden_channels_1, hidden_channels_2,
                 hidden_channels_3, hidden_channels_fc1, hidden_channels_fc2, out_channels):
        super(ProAffinity, self).__init__()
        self.protein_feature_extractor = ProteinFeatureExtractor(input_size, hidden_size_1, hidden_size_2)
        self.graph_conv_net = GraphConvNet(in_channels, hidden_channels_1, hidden_channels_2, hidden_channels_3)
        # 直接使用CoFormer默认参数初始化
        self.coformer = CoFormer()
        self.fc1 = torch.nn.Linear(Config.HIDDEN_CHANNELS_INPUT, hidden_channels_fc1)
        self.fc2 = torch.nn.Linear(hidden_channels_fc1, hidden_channels_fc2)
        self.fc3 = torch.nn.Linear(hidden_channels_fc2, out_channels)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, data_sequence, data_structure, data_interface_sequence, data_interface_structure,
                key_padding_mask=None, attn_mask=None):
        # 使用 ProteinFeatureExtractor 获取序列特征
        h1 = self.protein_feature_extractor(data_sequence)
        # 使用 GraphConvNet 获取图级别的特征
        h2 = self.graph_conv_net(data_structure)
        # 使用 CoFormer 进行特征提取
        h3 = self.coformer(data_interface_sequence, data_interface_structure)
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)
        h3 = self.dropout(h3)
        # 拼接所有特征
        h = torch.cat((h1, h2, h3), dim=0)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h
