import torch


class Config:
    # 数据路径
    TRA_DATA_ROOT1 = './training_dataset_esm'
    TRA_DATA_ROOT2 = './based_structure_training_dataset'
    TRA_DATA_ROOT3 = './residue_interface/esm2/training_dataset_esm'
    TRA_DATA_ROOT4 = './residue_interface/esmif/training_dataset_esmif'
    VAL_DATA_ROOT1 = './val_dataset_esm'
    VAL_DATA_ROOT2 = './based_structure_val_dataset'
    VAL_DATA_ROOT3 = './residue_interface/esm2/val_dataset_esm'
    VAL_DATA_ROOT4 = './residue_interface/esmif/val_dataset_esmif'
    # 模型参数
    INPUT_SIZE = 1280
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 128
    HIDDEN_CHANNELS_INPUT = 307200
    IN_CHANNELS = 40  # 节点特征维度
    HIDDEN_CHANNELS_1 = 256
    HIDDEN_CHANNELS_2 = 128
    HIDDEN_CHANNELS_3 = 64
    HIDDEN_CHANNELS_fc1 = 5120
    HIDDEN_CHANNELS_fc2 = 256
    OUT_CHANNELS = 1
    NUM_GAT_HEADS = 2
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.005
    DROPOUT_RATE = 0.5

    # 其他参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
