import torch
import os
import argparse
import numpy as np
import pandas as pd
from config.config import Config
from model.network.SSIFAffinity import SSIFAffinity
from tools.train import ProteinDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Protein Affinity Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file (.pth)')
    parser.add_argument('--affinity_csv', type=str, required=True,
                        help='Path to affinity CSV file containing PDBIDs')
    parser.add_argument('--esm_dir', type=str, required=True,
                        help='Directory containing ESM features')
    parser.add_argument('--structure_dir', type=str, required=True,
                        help='Directory containing structure features')
    parser.add_argument('--interface_seq_dir', type=str, required=True,
                        help='Directory containing interface sequence features')
    parser.add_argument('--interface_struct_dir', type=str, required=True,
                        help='Directory containing interface structure features')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV file path (default: predictions.csv)')
    return parser.parse_args()


def main(args):
    test_dataset = ProteinDataset(
        root1=args.esm_dir,
        root2=args.structure_dir,
        root3=args.interface_seq_dir,
        root4=args.interface_struct_dir,
        affinity_csv=args.affinity_csv,
        transform=None
    )

    device = torch.device(Config.DEVICE)
    model = SSIFffinity(
        Config.INPUT_SIZE,
        Config.HIDDEN_SIZE_1,
        Config.HIDDEN_SIZE_2,
        Config.IN_CHANNELS,
        Config.HIDDEN_CHANNELS_1,
        Config.HIDDEN_CHANNELS_2,
        Config.HIDDEN_CHANNELS_3,
        Config.HIDDEN_CHANNELS_fc1,
        Config.HIDDEN_CHANNELS_fc2,
        Config.OUT_CHANNELS
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    affinity_df = pd.read_csv(args.affinity_csv)
    pdb_ids = affinity_df.iloc[:, 0].tolist()

    predictions = []
    true_affinities = []

    with torch.no_grad():
        for seq_data, struct_data, interface_seq_data, interface_struct_data, affinity in test_dataset:
            seq_data = seq_data.to(device)
            struct_data = struct_data.to(device)
            interface_seq_data = interface_seq_data.to(device)
            interface_struct_data = interface_struct_data.to(device)

            output = model(seq_data, struct_data, interface_seq_data, interface_struct_data)
            predictions.append(output.cpu().numpy())
            true_affinities.append(affinity)

    results_df = pd.DataFrame({
        'PDBID': pdb_ids,
        'Predicted_Affinity': np.concatenate(predictions).flatten(),
        'True_Affinity': np.array(true_affinities).flatten()
    })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"预测结果已保存至: {args.output_csv}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    