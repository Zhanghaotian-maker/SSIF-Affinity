import torch
import esm
import matplotlib.pyplot as plt
import os
import pandas as pd


def process_sequence_features(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    for txt_filename in os.listdir(input_folder):
        if txt_filename.endswith('.txt'):
            txt_file_path = os.path.join(input_folder, txt_filename)

            data = []
            with open(txt_file_path, 'r') as file:
                file_content = file.read().strip()
                data.append(("protein" + txt_filename.replace(".txt", ""), file_content))

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            sequence_representations = []
            for i, (_, seq) in enumerate(data):
                sequence_tokens_representations = token_representations[i, 1:len(seq) + 1]
                sequence_representations.append(sequence_tokens_representations.cpu())

            tensor = sequence_representations[0]
            num_columns = tensor.shape[1]
            column_names = [f'col{i + 1}' for i in range(num_columns)]

            df = pd.DataFrame(tensor.numpy(), columns=column_names)
            csv_filename = txt_filename.replace(".txt", ".csv")
            full_csv_path = os.path.join(output_folder, csv_filename)
            df.to_csv(full_csv_path, index=False, header=False)

