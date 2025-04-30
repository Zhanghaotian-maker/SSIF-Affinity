from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import pandas as pd
import os
import string


class PDBProcessing:
    @staticmethod
    def extract_pdb_info(pdb_directory):
        file_data_list = []
        for filename in os.listdir(pdb_directory):
            if filename.endswith('.pdb'):
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                file_path = os.path.join(pdb_directory, filename)
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(filename, file_path)
                data = []
                chain_id_to_num = {char: idx for idx, char in enumerate(string.ascii_uppercase)}
                for model in structure:
                    for chain in model:
                        pp_builder = PPBuilder()
                        pp = pp_builder.build_peptides(chain)
                        first_char = chain.id[0].upper()
                        chain_num = chain_id_to_num.get(first_char, -1)
                        for polypeptide in pp:
                            for residue in polypeptide:
                                if residue.id[0] == 'H':
                                    continue
                                for atom in residue:
                                    row = {
                                        'Chain_name': chain.id,
                                        'Chain_num': chain_num,
                                        'Residue': residue.get_resname(),
                                        'Type': atom.name,
                                        'x': atom.coord[0],
                                        'y': atom.coord[1],
                                        'z': atom.coord[2]
                                    }
                                    data.append(row)
                file_data_list.append((csv_filename, pd.DataFrame(data)))
        return file_data_list

    @staticmethod
    def process_csv_files(file_data_list, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for csv_filename, df in file_data_list:
            if df.empty or not {'Type', 'Residue', 'x', 'y', 'z'}.issubset(df.columns):
                continue
            df['AtomType'] = df['Type']
            df['Atom'] = df['Type'].str[0]
            df['Res'] = df['Residue']
            df['X'] = df['x']
            df['Y'] = df['y']
            df['Z'] = df['z']
            new_columns = ['Chain_name', 'Chain_num', 'Res', 'AtomType', 'Atom', 'X', 'Y', 'Z']
            df_processed = df[new_columns]
            output_path = os.path.join(output_dir, csv_filename)
            df_processed.to_csv(output_path, index=False)
