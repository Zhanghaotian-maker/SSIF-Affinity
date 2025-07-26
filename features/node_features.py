import pickle
import pandas as pd
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm, trange
import random
import prettytable as pt
import math
import argparse


def one_hot_encode_single(element):
    """
    对给定的元素列表进行 One-hot 编码。

    参数:
    elements: 一个包含元素字符串的列表

    返回:
    一个二维 NumPy 数组，其中每行代表一个元素的 One-hot 编码。
    """
    # 定义元素和它们索引的映射
    element_mapping = {'C': 0,  'O': 1, 'N': 2,  'S': 3}

    # 初始化结果数组
    result = [0] * len(element_mapping)

    # 填充结果列表
    index = element_mapping.get(element, None)
    if index is not None:
        result[index] = 1

    return result


def one_hot_encode(residue):
    # 定义残基和它们索引的映射
    residue_mapping = {'R': 0, 'M': 1, 'V': 2, 'N': 3, 'P': 4, 'T': 5, 'F': 6, 'D': 7, 'I': 8,
                       'A': 9, 'G': 10, 'E': 11, 'L': 12, 'S': 13, 'K': 14, 'Y': 15, 'C': 16, 'H': 17, 'Q': 18, 'W': 19}

    # 初始化结果数组
    result = [0] * len(residue_mapping)

    # 填充结果列表
    index = residue_mapping.get(residue, None)
    if index is not None:
        result[index] = 1

    return result


def one_hot_encode_quality(quality):
    quality_mapping = {'hbond_acceptor': 0, 'hbond_donor':1, 'weak_hbond_donor': 2,
                        'pos_ionisable': 3, 'neg_ionisable':4, 'Hydrophobe': 5,
                        'carbonyl_oxygen': 6, 'carbonyl_carbon': 7, 'Aromatic': 8, 'else': 9}

    # 初始化结果数组
    result = [0] * len(quality_mapping)

    # 填充结果列表
    index = quality_mapping.get(quality, None)
    if index is not None:
        result[index] = 1

    return result


def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0,0,0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 1],
         'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1], 'CZ2': [0, 1, 1],
         'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features


def get_quality_category(residue, atom_type, quality_data):
    # 遍历quality_data中的所有键（即属性类别）
    for category, residue_map in quality_data.items():
        # 检查当前res是否在res_map中
        if residue in residue_map:
            # 检查atomtype是否在res对应的列表中
            if atom_type in residue_map[residue]:
                # 如果都满足，则返回当前的category
                return category
    # 如果没有找到匹配的category，则返回None或抛出一个异常
    return 'else'


def get_af(file_path, output_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    quality_data = {'hbond_acceptor': {'ALA': ['O', 'OXT'], 'ARG': ['O', 'OXT'], 'ASN': ['ND2', 'O', 'OD1', 'OXT'],
                                       'ASP': ['OD1', 'OD2', 'O', 'OXT'], 'CYS': ['SG', 'O', 'OXT'],
                                       'GLN': ['NE2', 'O', 'OE1', 'OXT'], 'GLU': ['OE1', 'OE2', 'O', 'OXT'],
                                       'GLY': ['O', 'OXT'], 'HIS': ['ND1', 'NE2', 'CE1', 'CD2', 'O', 'OXT'],
                                       'ILE': ['O', 'OXT'], 'LEU': ['O', 'OXT'], 'LYS': ['O', 'OXT'],
                                       'MET': ['SD', 'O', 'OXT'], 'PHE': ['O', 'OXT'], 'PRO': ['O', 'OXT'],
                                       'SER': ['OG', 'O', 'OXT'], 'THR': ['OG1', 'O', 'OXT'], 'TRP': ['O', 'OXT'],
                                       'TYR': ['OH', 'O', 'OXT'], 'VAL': ['O', 'OXT']},
                    'hbond_donor': {'ALA': ['N'], 'ARG': ['N', 'NE', 'NH1', 'NH2'], 'ASN': ['N', 'ND2', 'OD1'],
                                    'ASP': ['N'], 'CYS': ['N', 'SG'], 'GLN': ['N', 'NE2', 'OE1'], 'GLU': ['N'],
                                    'GLY': ['N'], 'HIS': ['N', 'ND1', 'CE1', 'NE2', 'CD2'], 'ILE': ['N'],
                                    'LEU': ['N'], 'LYS': ['N', 'NZ'], 'MET': ['N'], 'PHE': ['N'], 'SER': ['N', 'OG'],
                                    'THR': ['N', 'OG1'], 'TRP': ['N', 'NE1'], 'TYR': ['N', 'OH'], 'VAL': ['N']},
                    'weak_hbond_donor': {'ALA': ['CA', 'CB'], 'ARG': ['CA', 'CB', 'CG', 'CD'], 'ASN': ['CA', 'CB'],
                                         'ASP': ['CA', 'CB'], 'CYS': ['CA', 'CB'], 'GLN': ['CA', 'CB', 'CG'],
                                         'GLU': ['CA', 'CB', 'CG'], 'GLY': ['CA'], 'HIS': ['CA', 'CB'],
                                         'ILE': ['CA', 'CB', 'CG1', 'CD1', 'CG2'],
                                         'LEU': ['CA', 'CB', 'CG', 'CD1', 'CD2'],
                                         'LYS': ['CA', 'CB', 'CG', 'CD', 'CE'], 'MET': ['CA', 'CB', 'CG', 'CE'],
                                         'PHE': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                                         'PRO': ['CA', 'CB', 'CG', 'CD'], 'SER': ['CA', 'CB'],
                                         'THR': ['CA', 'CB', 'CG2'],
                                         'TRP': ['CA', 'CB', 'CD1', 'CE3', 'CZ3', 'CH2', 'CZ2'],
                                         'TYR': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                                         'VAL': ['CA', 'CB', 'CG1', 'CG2']},
                    'pos_ionisable': {'ARG': ['NE', 'CZ', 'NH1', 'NH2'], 'HIS': ['CG', 'ND1', 'CE1', 'NE2', 'CD2'],
                                      'LYS': ['NZ']},
                    'neg_ionisable': {'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']},
                    'Hydrophobe': {'ALA': ['CB'], 'ARG': ['CB', 'CG'], 'ASN': ['CB'], 'ASP': ['CB'], 'CYS': ['CB'],
                                   'GLN': ['CB', 'CG'], 'GLU': ['CB', 'CG'], 'HIS': ['CB'],
                                   'ILE': ['CB', 'CG1', 'CD1', 'CG2'],
                                   'LEU': ['CB', 'CG', 'CD1', 'CD2'], 'LYS': ['CB', 'CG', 'CD'],
                                   'MET': ['CB', 'CG', 'CE', 'SD'],
                                   'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'PRO': ['CB', 'CG'],
                                   'THR': ['CG2'],
                                   'TRP': ['CB', 'CG', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'],
                                   'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'], 'VAL': ['CB', 'CG1', 'CG2']},
                    'carbonyl_oxygen': {'ALA': ['O'], 'ARG': ['O'], 'ASN': ['O'], 'ASP': ['O'], 'CYS': ['O'],
                                        'GLN': ['O'], 'GLU': ['O'], 'GLY': ['O'], 'HIS': ['O'], 'ILE': ['O'],
                                        'LEU': ['O'], 'LYS': ['O'], 'MET': ['O'], 'PHE': ['O'], 'PRO': ['O'],
                                        'SER': ['O'], 'THR': ['O'], 'TRP': ['O'], 'TYR': ['O'], 'VAL': ['O']},
                    'carbonyl_carbon': {'ALA': ['C'], 'ARG': ['C'], 'ASN': ['C'], 'ASP': ['C'], 'CYS': ['C'],
                                        'GLN': ['C'], 'GLU': ['C'], 'GLY': ['C'], 'HIS': ['C'], 'ILE': ['C'],
                                        'LEU': ['C'], 'LYS': ['C'], 'MET': ['C'], 'PHE': ['C'], 'PRO': ['C'],
                                        'SER': ['C'], 'THR': ['C'], 'TRP': ['C'], 'TYR': ['C'], 'VAL': ['C']},
                    'Aromatic': {'HIS': ['CG', 'ND1', 'CE1', 'NE2', 'CD2'],
                                 'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                                 'TRP': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
                                 'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']}}
    atom_count = -1
    df = pd.read_csv(file_path)
    rows = []
    relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19,
                            'CO': 59, 'V': 51, 'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3,
                            'NA': 23, 'HG': 200.6, 'MN': 55, 'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9,
                            'SE': 79, 'NI': 58.7}

    for index, row in df.iterrows():  # 这里可以访问每一行的数据
        atom = row['Atom']
        residue = row['Res']
        encode1 = one_hot_encode_single(atom)
        atom_type = row['AtomType']
        category = get_quality_category(residue, atom_type, quality_data)
        one_hot = one_hot_encode_quality(category)
        if atom not in relative_atomic_mass.keys():
            continue
        atom_count += 1
        if row['AtomType'] not in ['N', 'CA', 'C', 'O', 'H']:
            is_sidechain = 1
        else:
            is_sidechain = 0
        res = res_dict[row['Res']]
        try:
            atom_fea = atom_fea_dict[res][atom_type]
        except KeyError:
            atom_fea = [0.5, 0.5, 0.5]
        encode2 = one_hot_encode(res)
        tmps = {
            'index': atom_count, 'atom_type': row['AtomType'], 'atom': row['Atom'], 'res': row['Res'],
            'x': row['X'], 'y': row['Y'], 'z': row['Z'],  'chain_index': row['Chain_num'],
            'element_typeC': encode1[0], 'element_typeO': encode1[1], 'element_typeN': encode1[2],
            'element_typeS': encode1[3],  'residue_typeR': encode2[0], 'residue_typeM': encode2[1],
            'residue_typeV': encode2[2], 'residue_typeN': encode2[3], 'residue_typeP': encode2[4],
            'residue_typeT': encode2[5], 'residue_typeF': encode2[6], 'residue_typeD': encode2[7],
            'residue_typeI': encode2[8], 'residue_typeA': encode2[9], 'residue_typeG': encode2[10],
            'residue_typeE': encode2[11], 'residue_typeL': encode2[12], 'residue_typeS': encode2[13],
            'residue_typeK': encode2[14], 'residue_typeY': encode2[15], 'residue_typeC': encode2[16],
            'residue_typeH': encode2[17], 'residue_typeQ': encode2[18], 'residue_typeW': encode2[19],
            'hbond_acceptor': one_hot[0], 'hbond_donor': one_hot[1], 'weak_hbond_donor': one_hot[2],
            'pos_ionisable': one_hot[3], 'neg_ionisable': one_hot[4], 'Hydrophobe': one_hot[5],
            'carbonyl_oxygen': one_hot[6], 'carbonyl_carbon': one_hot[7], 'Aromatic': one_hot[8], 'else': one_hot[9],
            'mass': relative_atomic_mass[atom], 'is_sidechain': is_sidechain,
            'charge': atom_fea[0], 'num_H': atom_fea[1], 'ring': atom_fea[2]
        }
        rows.append(tmps)
    af = pd.DataFrame(rows)
    af_df = pd.DataFrame(af)
    af_df.to_csv(output_path, index=False)


def process_directory(path, prefix):
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            output_path = os.path.join(prefix, f"{filename}")
            get_af(file_path, output_path)
            




