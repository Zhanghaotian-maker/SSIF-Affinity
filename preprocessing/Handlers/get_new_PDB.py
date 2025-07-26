import os
import pandas as pd


class GetNewPDB:
    @staticmethod
    def process_file(input_path, output_dir):
        try:
            df = pd.read_csv(input_path)
            required_cols = ['Atom_id', 'Type', 'Chain_name', 'Chain_num', 'Residue', 'x_new', 'y_new', 'z_new']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return
            df['AtomType'] = df['Type']
            df['Atom'] = df['Type'].str[0]
            df['Res'] = df['Residue']
            df['X'] = df['x_new'].round(3)
            df['Y'] = df['y_new'].round(3)
            df['Z'] = df['z_new'].round(3)
            df.insert(0, 'ATOM_HEADER', 'ATOM')
            df_sorted = df.sort_values(by='Atom_id')
            if not df_sorted.empty and len(df_sorted) >= 2:
                counter = 1
                df_sorted['Counter'] = 1
                prev_chain = df_sorted.iloc[0]['Chain_name']
                for idx in range(1, len(df_sorted)):
                    current_chain = df_sorted.iloc[idx]['Chain_name']
                    if current_chain != prev_chain:
                        counter += 1
                        prev_chain = current_chain
                    df_sorted.at[idx, 'Counter'] = counter
            pdb_filename = os.path.splitext(os.path.basename(input_path))[0] + '.pdb'
            pdb_path = os.path.join(output_dir, pdb_filename)
            with open(pdb_path, 'w') as f:
                for _, row in df_sorted.iterrows():
                    atom_header = str(row['ATOM_HEADER']).ljust(6)
                    atom_id = str(row['Atom_id']).rjust(5)
                    atom_type = str(row['AtomType']).ljust(4)
                    res_name = str(row['Res']).ljust(3)
                    chain = str(row['Chain_name']).ljust(1)
                    counter = str(row['Counter']).rjust(4)
                    x = f"{float(row['X']):8.3f}"
                    y = f"{float(row['Y']):8.3f}"
                    z = f"{float(row['Z']):8.3f}"
                    element = str(row['Atom']).rjust(2)
                    pdb_line = (
                        f"{atom_header}"
                        f"{atom_id} "
                        f"{atom_type}"
                        f" {res_name}"
                        f" {chain}"
                        f"{counter}    "
                        f"{x}{y}{z}"
                        "  1.00  0.00      "
                        f"      {element}\n"
                    )
                    f.write(pdb_line)
        except Exception as e:
            print(f"处理文件出错: {str(e)}")
            
            