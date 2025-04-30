import torch
import torch.nn.functional as f
from config.config import Config
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset
from torch_geometric.nn.conv import MessagePassing


class EdgeFeatureConv(MessagePassing):
    def __init__(self, in_channels, in_channels_edge, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')
        self.lin_node = torch.nn.Linear(in_channels, out_channels)
        self.lin_edge = torch.nn.Linear(in_channels_edge, out_channels)
        self.lin_combined = torch.nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        combined = torch.cat([x_j, edge_attr], dim=1)
        return self.lin_combined(combined)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.lin_node.in_features}, {self.lin_node.out_features})'


class GraphNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, hidden_channels_3,
                 heads=2, dropout_rate=0.5, concat=False):
        super(GraphNet, self).__init__()

        # Edge Feature Convolution Layers
        self.conv1 = EdgeFeatureConv(in_channels, 5, hidden_channels_1)
        self.conv2 = EdgeFeatureConv(hidden_channels_1, 5, hidden_channels_2)
        self.conv3 = EdgeFeatureConv(hidden_channels_2, 5, hidden_channels_3)

        # Graph Attention Network Layer
        self.gat_conv = GATConv(hidden_channels_3, hidden_channels_3, heads=heads, dropout=dropout_rate,
                                add_self_loops=False, concat=concat)
        # Dropout Layer
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data_structure):
        x, edge_index, edge_attr = data_structure.x, data_structure.edge_index, data_structure.edge_attr
        x = x.to(torch.float32)
        h = x
        h = self.conv1(h, edge_index, edge_attr)
        h = f.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = f.relu(h)
        h = self.dropout(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = f.relu(h)
        h = self.dropout(h)

        h = self.gat_conv(h, edge_index)
        h = self.dropout(h)
        h = torch.flatten(h)
        
        return h

