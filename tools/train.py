import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data
from config.config import Config
from torch.utils.data import Dataset
from model.network.SSIFAffinity import SSIFAffinity
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
    def __init__(self, root1, root2, root3, root4, affinity_csv, transform=None):
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        self.root4 = root4
        self.affinity_df = pd.read_csv(affinity_csv)
        self.transform = transform
        self.filenames = self.affinity_df['ID'].values
        self.affinities = self.affinity_df['affinity'].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        affinity = self.affinities[idx]

        # 处理 data_sequence
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

        # 处理 data_structure
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
                neighbor = neighbor_info[0]
                attrs = neighbor_info[1:]
                edge_index.append([int(node), int(neighbor)])
                edge_attr.append(attrs)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x2 = node_features_tensor
        x2 = sample_or_pad_nodes(x2, num_nodes=2000)
        data_structure = Data(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        if self.transform is not None:
            data_structure = self.transform(data_structure)

        # 处理 data_interface_sequence
        interface_seq_path = os.path.join(self.root3, f"{file_name}.csv")
        df_interface_seq = pd.read_csv(interface_seq_path, header=None)
        num_rows_interface_seq = len(df_interface_seq)

        if num_rows_interface_seq > 400:
            start_row_interface_seq = (num_rows_interface_seq - 400) // 2
            df_interface_seq = df_interface_seq.iloc[start_row_interface_seq:start_row_interface_seq + 400]
        else:
            zero_row_interface_seq = [0] * 1280
            num_to_add_interface_seq = 400 - num_rows_interface_seq
            df_interface_seq = pd.concat(
                [df_interface_seq, pd.DataFrame([zero_row_interface_seq] * num_to_add_interface_seq)],
                ignore_index=True
            )

        interface_seq_tensor = torch.tensor(df_interface_seq.values, dtype=torch.float)
        data_interface_sequence = Data(x=interface_seq_tensor)
        if self.transform is not None:
            data_interface_sequence = self.transform(data_interface_sequence)

        # 处理 data_interface_structure
        interface_str_path = os.path.join(self.root4, f"{file_name}.csv")
        df_interface_str = pd.read_csv(interface_str_path, header=None)
        num_rows_interface_str = len(df_interface_str)

        if num_rows_interface_str > 400:
            start_row_interface_str = (num_rows_interface_str - 400) // 2
            df_interface_str = df_interface_str.iloc[start_row_interface_str:start_row_interface_str + 400]
        else:
            zero_row_interface_str = [0] * 512
            num_to_add_interface_str = 400 - num_rows_interface_str
            df_interface_str = pd.concat(
                [df_interface_str, pd.DataFrame([zero_row_interface_str] * num_to_add_interface_str)],
                ignore_index=True
            )

        interface_str_tensor = torch.tensor(df_interface_str.values, dtype=torch.float)
        data_interface_structure = Data(x=interface_str_tensor)
        if self.transform is not None:
            data_interface_structure = self.transform(data_interface_structure)

        return data_sequence, data_structure, data_interface_sequence, data_interface_structure, affinity


train_Affinity_csv = './training_affinity.csv'
val_Affinity_csv = './val_affinity.csv'


def train_model():
    # 加载数据集
    dataset = ProteinDataset(root1=Config.TRA_DATA_ROOT1, root2=Config.TRA_DATA_ROOT2, root3=Config.TRA_DATA_ROOT3,
                             root4=Config.TRA_DATA_ROOT4, affinity_csv=train_Affinity_csv,
                             transform=None)
    val_dataset = ProteinDataset(root1=Config.VAL_DATA_ROOT1, root2=Config.VAL_DATA_ROOT2, root3=Config.VAL_DATA_ROOT3,
                                 root4=Config.VAL_DATA_ROOT4, affinity_csv=val_Affinity_csv,
                                 transform=None)

    # 初始化模型
    model = SSIF-Affinity(Config.INPUT_SIZE, Config.HIDDEN_SIZE_1, Config.HIDDEN_SIZE_2, Config.IN_CHANNELS,
                          Config.HIDDEN_CHANNELS_1, Config.HIDDEN_CHANNELS_2, Config.HIDDEN_CHANNELS_3,
                          Config.HIDDEN_CHANNELS_fc1, Config.HIDDEN_CHANNELS_fc2, Config.OUT_CHANNELS).to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # ===== 添加学习率调度器和早停机制 =====
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控验证损失最小化
        factor=0.5,  # 学习率降低因子
        patience=5,  # 连续3个epoch验证损失未改善则降低学习率
        verbose=True,  # 打印学习率变化信息
        min_lr=1e-6  # 最小学习率
    )

    # 早停机制参数
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 20
    # 创建结果目录结构
    result_dir = './result'
    best_model_dir = os.path.join(result_dir, 'best_models')
    all_models_dir = os.path.join(result_dir, 'all_models')
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(all_models_dir, exist_ok=True)

    best_model_path = os.path.join(best_model_dir, 'best_model.pth')

    # 训练历史记录
    train_history = []
    val_history = []

    # 训练循环
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0

        batch_size = Config.BATCH_SIZE
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [dataset[idx] for idx in batch_indices]

            batch_data = [
                (
                    data_sequence.to(Config.DEVICE),
                    data_structure.to(Config.DEVICE),
                    data_interface_sequence.to(Config.DEVICE),
                    data_interface_structure.to(Config.DEVICE),
                    torch.tensor([affinity], dtype=torch.float).to(Config.DEVICE)
                )
                for data_sequence, data_structure, data_interface_sequence, data_interface_structure, affinity in
                batch_data
            ]

            optimizer.zero_grad()
            total_batch_loss = 0
            batch_samples = len(batch_data)

            for data_sequence, data_structure, data_interface_sequence, data_interface_structure, affinity in batch_data:
                out = model(data_sequence, data_structure, data_interface_sequence, data_interface_structure)
                loss = criterion(out.squeeze(-1), affinity.squeeze(-1))
                loss.backward()
                total_batch_loss += loss.item() * affinity.size(0)

            optimizer.step()

            total_loss += total_batch_loss
            total_samples += batch_samples

        avg_train_loss = total_loss / total_samples if total_samples > 0 else 0
        train_history.append(avg_train_loss)

        # ===== 验证过程 - 手动实现批次循环 =====
        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            val_indices = list(range(len(val_dataset)))

            for i in range(0, len(val_dataset), batch_size):
                batch_indices = val_indices[i:i + batch_size]
                batch_data = [val_dataset[idx] for idx in batch_indices]

                batch_data = [
                    (
                        data_sequence.to(Config.DEVICE),
                        data_structure.to(Config.DEVICE),
                        data_interface_sequence.to(Config.DEVICE),
                        data_interface_structure.to(Config.DEVICE),
                        torch.tensor([affinity], dtype=torch.float).to(Config.DEVICE)
                    )
                    for data_sequence, data_structure, data_interface_sequence, data_interface_structure, affinity in
                    batch_data
                ]

                for data_sequence, data_structure, data_interface_sequence, data_interface_structure, affinity in batch_data:
                    out = model(data_sequence, data_structure, data_interface_sequence, data_interface_structure)
                    loss = criterion(out.squeeze(-1), affinity.squeeze(-1))
                    val_loss += loss.item() * affinity.size(0)
                    val_samples += affinity.size(0)

        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        val_history.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f},'
            f' LR: {optimizer.param_groups[0]["lr"]:.7f}')
        epoch_model_path = os.path.join(all_models_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"模型已保存到: {epoch_model_path}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': best_val_loss,
            }, best_model_path)

            print(f"最佳模型已更新 (epoch: {best_epoch}, 验证损失: {best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证损失未改善 ({avg_val_loss:.6f} > {best_val_loss:.6f})，早停计数器: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"早停于第{epoch + 1}轮，验证损失连续{PATIENCE}轮未改善")
            break

        print(f"训练完成，最佳验证损失: {best_val_loss:.6f} (epoch: {best_epoch})")

