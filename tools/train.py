import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from config.config import Config
from torch.utils.data import Dataset
from model.network.SSF_Affinity import SSFAffinity
import pandas as pd
import json
import os
import numpy as np


def sample_or_pad_nodes(x, num_nodes=2000):
    num_existing_nodes = x.size(0)
    if num_existing_nodes > num_nodes:
        indices = torch.randperm(num_existing_nodes)[:num_nodes]
        x = x[indices]
    elif num_existing_nodes < num_nodes:
        zero_padding = torch.zeros(num_nodes - num_existing_nodes, x.size(1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, zero_padding], dim=0)
    return x


class ProteinDataset(Dataset):
    def __init__(self, root1, root2, affinity_csv, transform=None):
        self.root1 = root1
        self.root2 = root2
        self.affinity_df = pd.read_csv(affinity_csv)
        self.transform = transform
        self.filenames = self.affinity_df['ID'].values  
        self.affinities = self.affinity_df['affinity'].values 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        affinity = self.affinities[idx]

        csv1_path = os.path.join(self.root1, f"{file_name}_0.csv")
        csv2_path = os.path.join(self.root1, f"{file_name}_1.csv")

        df1 = pd.read_csv(csv1_path, header=None)  
        df2 = pd.read_csv(csv2_path, header=None)  
        combined_df = pd.concat([df1, df2], ignore_index=True)
        num_rows = len(combined_df)
        if num_rows > 2000:
            start_row = (num_rows - 2000) // 2
            combined_df = combined_df.iloc[start_row:start_row + 2000]
        else:
            zero_row = [0] * 1280  
            num_to_add = 2000 - num_rows

            upper_pad = num_to_add // 2
            lower_pad = num_to_add - upper_pad

            combined_df = pd.concat(
                [pd.DataFrame([zero_row] * upper_pad), combined_df, pd.DataFrame([zero_row] * lower_pad)],
                ignore_index=True)
        combined_tensor = torch.tensor(combined_df.values, dtype=torch.float)
        x1 = combined_tensor
        data_sequence = Data(x=x1)
        if self.transform is not None:
            data_sequence = self.transform(data_sequence)
        # 获得结构数据
        # 读取节点特征
        node_features_csv = os.path.join(self.root2, f"{file_name}.csv")
        node_features = pd.read_csv(node_features_csv, header=0).iloc[:, 7:47].values  # 假设节点特征在8-47列
        node_features_tensor = torch.tensor(node_features)
        # 读取边信息
        edge_info_json = os.path.join(self.root2, f"{file_name}.json")

        with open(edge_info_json, 'r') as f:
            edge_info = json.load(f)['neighbor_map']

        edge_index = []
        edge_attr = []

        for node, neighbors in edge_info.items():
            for neighbor_info in neighbors:
                neighbor = neighbor_info[0]  # 假设第一个元素是邻居节点ID
                attrs = neighbor_info[1:]  # 剩余元素是边特征
                edge_index.append([int(node), int(neighbor)])
                edge_attr.append(attrs)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x2 = node_features_tensor
        x2 = sample_or_pad_nodes(x2, num_nodes=2000)
        data_structure = Data(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        if self.transform is not None:
            data_structure = self.transform(data_structure)

        return data_sequence, data_structure, affinity


train_Affinity_csv = '../training_affinity.csv'
val_Affinity_csv = '../val_affinity.csv'


def train_model():
    # 加载数据集
    dataset = ProteinDataset(root1=Config.TRA_DATA_ROOT1, root2=Config.TRA_DATA_ROOT2, affinity_csv=train_Affinity_csv,
                             transform=None)
    val_dataset = ProteinDataset(root1=Config.VAL_DATA_ROOT1, root2=Config.VAL_DATA_ROOT2, affinity_csv=val_Affinity_csv
                                 , transform=None)
    # 初始化模型
    model = SSFAffinity(Config.INPUT_SIZE, Config.HIDDEN_SIZE_1, Config.HIDDEN_SIZE_2, Config.IN_CHANNELS, 
                        Config.HIDDEN_CHANNELS_1, Config.HIDDEN_CHANNELS_2, Config.HIDDEN_CHANNELS_3, 
                        Config.HIDDEN_CHANNELS_fc1, Config.HIDDEN_CHANNELS_fc2,
                        Config.OUT_CHANNELS).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    model_paths = []
    model.train()
    for epoch in range(Config.EPOCHS):
        result_dir = '../result'
        total_loss = 0

        batch_size = Config.BATCH_SIZE
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [dataset[idx] for idx in batch_indices]
            batch_data = [(data_sequence.to(Config.DEVICE), data_structure.to(Config.DEVICE), 
                           torch.tensor([affinity], dtype=torch.float).to(Config.DEVICE))
                          for data_sequence, data_structure, affinity in batch_data]

            optimizer.zero_grad()
            total_batch_loss = 0

            for data_sequence, data_structure, affinity in batch_data:
                data_sequence = data_sequence.to(Config.DEVICE)
                data_structure = data_structure.to(Config.DEVICE)
                out = model(data_sequence, data_structure)
                loss = criterion(out.squeeze(-1), affinity.squeeze(-1))  # 将affinity转换为Tensor
                loss.backward()
                total_batch_loss += loss.item()

            # 在整个批次后更新权重
            optimizer.step()

            total_loss += total_batch_loss

        # 验证过程
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():  # 关闭梯度计算
            for val_data_sequence, val_data_structure, val_affinity in val_dataset:
                val_data_sequence = val_data_sequence.to(Config.DEVICE)
                val_data_structure = val_data_structure.to(Config.DEVICE)
                val_affinity = torch.tensor([val_affinity], dtype=torch.float).to(Config.DEVICE)
                val_output = model(val_data_sequence, val_data_structure)
                val_loss += criterion(val_output.squeeze(-1), val_affinity.squeeze(-1)).item()
        val_loss /= len(val_dataset)

        # 检查是否是最佳模型
        if val_loss < best_loss:
            best_loss = val_loss

        epoch_model_path = os.path.join(result_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        model_paths.append(epoch_model_path)

        print(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Train Loss: {total_loss / (len(dataset))}, Val Loss: {val_loss}')

        # 重置模型为训练模式
        model.train()
        print("模型已保存。")
    return model_paths
