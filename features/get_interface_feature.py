import os
import pandas as pd
from pathlib import Path


def process_raw_csv(source_dir, output_dir):
    """
    处理原始CSV文件（来自strcuture_end.py）
    1. 添加Index列（从1开始）
    2. 添加Residue_Index列（基于N原子递增）
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(source_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(source_dir, file_name)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"无法读取原始CSV文件 {file_name}: {e}")
                continue

            # 添加Index列
            df.insert(0, "Index", range(1, len(df) + 1))

            # 添加Residue_Index列
            if "Residue" not in df.columns:
                print(f"文件 {file_name} 缺少 Residue 列，跳过")
                continue

            df.insert(df.columns.get_loc("Residue") + 1, "Residue_Index", 0)
            residue_index = 0
            for index, row in df.iterrows():
                if row["Type"] == "N":
                    residue_index += 1
                df.at[index, "Residue_Index"] = residue_index

            # 保存处理后的文件
            output_path = os.path.join(output_dir, file_name)
            df.to_csv(output_path, index=False)
            print(f"已处理 {file_name}，保存到 {output_path}")


def merge_csv_files(source_dir, output_dir):
    """
    合并成对的CSV文件（来自merge.py）
    将同一前缀的1/2文件合并（如ABC1.csv和ABC2.csv -> ABC.csv）
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

    # 按前缀分组（去除末尾的1/2）
    file_groups = {}
    for file in csv_files:
        stem = Path(file).stem
        prefix = stem[:-1]
        suffix = stem[-1]

        if suffix in ['1', '2']:
            if prefix not in file_groups:
                file_groups[prefix] = {'1': None, '2': None}
            file_groups[prefix][suffix] = file

    # 合并每组文件
    for prefix, files in file_groups.items():
        file1 = files.get('1')
        file2 = files.get('2')

        if not file1 or not file2:
            continue

        try:
            df1 = pd.read_csv(os.path.join(source_dir, file1), header=None)
            df2 = pd.read_csv(os.path.join(source_dir, file2), header=None)

            if df1.shape[1] != df2.shape[1]:
                print(f"列数不匹配: {file1}({df1.shape[1]}) 和 {file2}({df2.shape[1]})")
                continue

            merged_df = pd.concat([df1, df2], ignore_index=True)
            output_file = os.path.join(output_dir, prefix[:-1] + ".csv")
            merged_df.to_csv(output_file, index=False, header=False)
            print(f"已合并 {file1} + {file2} -> {output_file}")

        except Exception as e:
            print(f"合并错误 {file1}+{file2}: {e}")


def process_files(binding_box_dir, raw_data_dir, esm_data_dir, output_dir):
    """
    从ESM数据中提取结合位点对应的残基（来自filter_align.py）
    1. 从binding_box获取Atom_id
    2. 从raw_data获取对应的Residue_Index
    3. 从ESM数据提取对应残基的特征
    """
    os.makedirs(output_dir, exist_ok=True)

    for binding_file in os.listdir(binding_box_dir):
        if not binding_file.endswith('.csv'):
            continue

        # 读取结合位点数据
        binding_path = os.path.join(binding_box_dir, binding_file)
        try:
            binding_df = pd.read_csv(binding_path)
            atom_ids = sorted(binding_df['Atom_id'].dropna().unique())
        except Exception as e:
            print(f"结合位点文件错误 {binding_file}: {e}")
            continue

        # 读取原始数据获取残基索引
        raw_path = os.path.join(raw_data_dir, binding_file)
        if not os.path.exists(raw_path):
            print(f"原始数据缺失 {binding_file}")
            continue

        try:
            raw_df = pd.read_csv(raw_path)
            residue_indices = set()

            for atom_id in atom_ids:
                match_rows = raw_df[raw_df.iloc[:, 0] == atom_id]
                if not match_rows.empty:
                    residue_idx = match_rows['Residue_Index'].iloc[0] if 'Residue_Index' in raw_df.columns else \
                    match_rows.iloc[0, 3]
                    residue_indices.add(residue_idx)

            residue_indices = sorted(residue_indices)
        except Exception as e:
            print(f"原始数据处理错误 {binding_file}: {e}")
            continue

        # 从ESM数据提取特征
        esm_path = os.path.join(esm_data_dir, binding_file)
        if not os.path.exists(esm_path):
            print(f"ESM数据缺失 {binding_file}")
            continue

        try:
            esm_df = pd.read_csv(esm_path, header=None)
            extracted_rows = [esm_df.iloc[idx] for idx in residue_indices if 0 <= idx < len(esm_df)]

            if extracted_rows:
                output_path = os.path.join(output_dir, binding_file)
                pd.concat(extracted_rows, axis=1).T.to_csv(output_path, header=False, index=False)
                print(f"已提取 {binding_file}: {len(extracted_rows)}个残基")
            else:
                print(f"无有效残基 {binding_file}")

        except Exception as e:
            print(f"ESM处理错误 {binding_file}: {e}")


if __name__ == "__main__":
    # ===== 第一步：预处理原始数据 =====
    process_raw_csv(
        source_dir='./dataset/raw_csv',
        output_dir='./dataset/raw_end_csv'
    )

    # ===== 第二步：合并ESM特征文件 =====
    # 序列特征合并
    merge_csv_files(
        source_dir='./sequence_dataset_esm',
        output_dir='./sequence_dataset_esm_merged'
    )

    # 结构特征合并
    merge_csv_files(
        source_dir='./structure_dataset_esmif',
        output_dir='./structure_dataset_esmif_merged'
    )

    # ===== 第三步：提取结合位点特征 =====
    # 处理序列特征
    process_files(
        binding_box_dir='./dataset/the_end_csv',
        raw_data_dir='./dataset/raw_end_csv',
        esm_data_dir='./sequence_dataset_esm_merged',
        output_dir='./residue_interface/esm2/sequence_dataset_esm'
    )

    # 处理结构特征
    process_files(
        binding_box_dir='./dataset/the_end_csv',
        raw_data_dir='./dataset/raw_end_csv',
        esm_data_dir='./structure_dataset_esmif_merged',
        output_dir='./residue_interface/esmif/structure_dataset_esmif'
    )

    print("=" * 50)
    print("所有处理步骤已完成！")
    print("=" * 50)
    
    