import esm.inverse_folding
import biotite.structure
import csv
import numpy as np
import os

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()

csv_file_path = './Complexes_chains.csv'
pdb_directory = './datasets/PDBs'
output_csv_file_path = './structure_dataset_esmif'

with open(csv_file_path, mode='r', newline='', encoding='UTF-8-SIG') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pdb_id = row['pdb_id']
        protein_1_str = row['Protein_1']
        protein_1_chars = [char for char in protein_1_str]

        protein_2_str = row['Protein_2']
        protein_2_chars = [char for char in protein_2_str]

        chain_ids = protein_1_chars + protein_2_chars
        pdb_file_path = os.path.join(pdb_directory, f"{pdb_id}.pdb")

        if os.path.exists(pdb_file_path):
            structure = esm.inverse_folding.util.load_structure(pdb_file_path, chain_ids)
            coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

            all_reps = []
            for target_chain_id in chain_ids:
                rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(
                    model, alphabet, coords, target_chain_id)
                all_reps.append(rep)

            combined_rep = np.vstack([rep.detach().cpu().numpy() for rep in all_reps])
            output_csv_file_path = os.path.join(pdb_directory, f"{pdb_id}.csv")
            with open(output_csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                for row in combined_rep:
                    writer.writerow(row)

            print(f"Combined representation for {pdb_id} saved to {output_csv_file_path}")
        else:
            print(f"PDB file {pdb_file_path} does not exist")


