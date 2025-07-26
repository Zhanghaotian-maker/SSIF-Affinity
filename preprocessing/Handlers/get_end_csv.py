import os
import pandas as pd


class GetEndCSV:
    @staticmethod
    def process_pdb_to_final_csv(pdb_dir, csv_invert_pdb_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(pdb_dir):
            if not filename.endswith('.pdb'):
                continue
            pdb_path = os.path.join(pdb_dir, filename)
            csv_invert_pdb_path = os.path.join(csv_invert_pdb_dir, os.path.splitext(filename)[0] + '.csv')
            final_csv_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.csv')
            try:
                pdb_data = pd.read_csv(pdb_path, delim_whitespace=True, header=None)
                pdb_data = pdb_data.drop(columns=[4, 9, 10])
                pdb_data.insert(3, 'Atom', pdb_data.pop(11))
                pdb_data.columns = ['Level', 'Atom_id', 'AtomType', 'Atom', 'Res', 'Chain_num', 'X', 'Y', 'Z']
                if os.path.exists(csv_invert_pdb_path):
                    csv_invert_pdb_df = pd.read_csv(csv_invert_pdb_path)
                    if 'Chain_num' not in pdb_data.columns:
                        pdb_data['Chain_num'] = None
                    for index, row in pdb_data.iterrows():
                        x, y, z = row['X'], row['Y'], row['Z']
                        matched_rows = csv_invert_pdb_df[
                            (csv_invert_pdb_df['X'] == x) & (csv_invert_pdb_df['Y'] == y) & (csv_invert_pdb_df['Z'] == z)]
                        if not matched_rows.empty:
                            pdb_data.at[index, 'Chain_num'] = matched_rows.iloc[0]['Chain_num']
                pdb_data.to_csv(final_csv_path, index=False)
            except Exception as e:
                print(f"处理文件出错: {str(e)}")
                
                
        