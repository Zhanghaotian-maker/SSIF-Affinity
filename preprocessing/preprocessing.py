import os
from Handlers.PDB_Processing import PDBProcessing
from Handlers.select_atoms import SelectAtoms
from Handlers.get_new_PDB import GetNewPDB
from Handlers.structure_distance import StructureDistance
from Handlers.get_end_CSV import GetEndCSV


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_dir = os.path.join(script_dir, '../datasets/PDBs')
    csv_dir = os.path.join(script_dir, '../datasets/CSVs')
    end_pdb_dir = os.path.join(script_dir, '../datasets/the_end_PDB')
    end_csv_dir = os.path.join(script_dir, '../datasets/the_end_CSV')
    executable_path = '../features/get_features.exe'

    # 步骤1：从PDB文件提取信息到CSV文件
    file_data = PDBProcessing.extract_pdb_info(pdb_dir)
    PDBProcessing.process_csv_files(file_data, csv_dir)

    # 步骤2：确定结合区域，并选定原子
    SelectAtoms.analyze_csv_files(csv_dir)

    # 步骤3：被选定原子得到CSV文件转换为PDB文件用于边距离的生成
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(csv_dir, filename)
            GetNewPDB.process_file(input_path, end_pdb_dir)

    # 步骤4：处理PDB文件（调用外部程序）
    StructureDistance.process_structure_distance(end_pdb_dir, end_pdb_dir, executable_path)

    # 步骤5：生成最终CSV文件
    GetEndCSV.process_pdb_to_final_csv(end_pdb_dir, csv_dir, end_csv_dir)


if __name__ == "__main__":
    main()
