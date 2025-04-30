import torch
import torch.nn as nn
from config.config import Config
from model.modules.feature_CNN_LSTM import ProteinFeatureExtractor
from model.modules.feature_GNN import GraphNet


class SSFAffinity(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, in_channels, hidden_channels_1, hidden_channels_2, 
                 hidden_channels_3, hidden_channels_fc1, hidden_channels_fc2, out_channels):
        super(SSFAffinity, self).__init__()
        self.protein_feature_extractor = ProteinFeatureExtractor(input_size, hidden_size_1, hidden_size_2)
        self.graph_net = GraphNet(in_channels, hidden_channels_1, hidden_channels_2, hidden_channels_3)
        self.fc1 = torch.nn.Linear(Config.HIDDEN_CHANNELS_INPUT, hidden_channels_fc1)
        self.fc2 = torch.nn.Linear(hidden_channels_fc1, hidden_channels_fc2)
        self.fc3 = torch.nn.Linear(hidden_channels_fc2, out_channels)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, data_sequence, data_structure):

        h1 = self.protein_feature_extractor(data_sequence)
        h2 = self.graph_conv_net(data_structure)
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)
        h = torch.cat((h1, h2), dim=0)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h
    