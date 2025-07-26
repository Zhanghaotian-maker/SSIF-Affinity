import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import string


def extract_pdb_to_csv(pdb_directory):
    """从PDB文件中提取信息并保存为CSV文件"""
    for filename in os.listdir(pdb_directory):
        if filename.endswith('.pdb'):
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
                                    'Atom_id': f"{chain.id}_{residue.id[1]}_{atom.name}",
                                    'Chain_name': chain.id,
                                    'Chain_num': chain_num,
                                    'Residue': residue.get_resname(),
                                    'Type': atom.name,
                                    'x': atom.coord[0],
                                    'y': atom.coord[1],
                                    'z': atom.coord[2]
                                }
                                data.append(row)

            df = pd.DataFrame(data)
            df['x_new'] = df['x']
            df['y_new'] = df['y']
            df['z_new'] = df['z']
            output_directory = os.path.join(os.path.dirname(pdb_directory), 'raw_csv')
            os.makedirs(output_directory, exist_ok=True)
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            df.to_csv(os.path.join(output_directory, csv_filename), index=False)
            print(f"已提取并保存: {csv_filename}")


def process_csv_files(directory):
    """处理CSV文件，规范列名并保存到新目录"""
    new_folder_path = os.path.join(os.path.dirname(directory), 'raw_csv')
    os.makedirs(new_folder_path, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            required_columns = ['Atom_id', 'Type', 'Residue', 'x_new', 'y_new', 'z_new']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告: 跳过 {filename}，缺少列: {missing_columns}")
                continue

            df['AtomType'] = df['Type']
            df['Atom'] = df['Type'].str[0]
            df['Res'] = df['Residue']
            df['X'] = df['x_new']
            df['Y'] = df['y_new']
            df['Z'] = df['z_new']

            new_columns = ['Atom_id', 'AtomType', 'Atom', 'Res', 'X', 'Y', 'Z']
            df = df[new_columns]

            new_filename = os.path.join(new_folder_path, filename)
            df.to_csv(new_filename, index=False)
            print(f"已处理并保存: {new_filename}")


if __name__ == "__main__":
    pdb_directory = './datasets/PDBs'

    print("开始从PDB文件中提取信息...")
    extract_pdb_to_csv(pdb_directory)
    print("PDB信息提取完成!")

    print("\n开始处理CSV文件...")
    process_csv_files(pdb_directory)
    print("CSV文件处理完成!")

    print("\n所有处理已完成!")
    