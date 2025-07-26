import csv, pickle, json, os
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict as ddict


def createSortedNeighbors(contacts, bonds, max_neighbors):
    bond_true = 1
    bond_false = 0
    neighbor_map = ddict(list)
    dtype = [('index2', int), ('distance', float), ('x1', float), ('y1', float), ('z1', float),
             ('bool_bond', int)]
    idx = 0
    for contact in contacts:
        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5],
                                            bond_true))
            neighbor_map[contact[1]].append((contact[0],  contact[2], contact[6], contact[7], contact[8],
                                             bond_true))
        else:
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5],
                                             bond_false))
            neighbor_map[contact[1]].append((contact[0], contact[2], contact[6], contact[7], contact[8],
                                             bond_false))
        idx += 1

    for k, v in neighbor_map.items():
        if len(v) < max_neighbors:
            true_nbrs = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[0:len(v)]
            true_nbrs.extend([(0, 0, 0, 0, 0, 0) for _ in range(max_neighbors - len(v))])
            neighbor_map[k] = true_nbrs
        else:
            neighbor_map[k] = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[
                              0:max_neighbors]

    return neighbor_map


def process_json_directory(input_dir, output_dir, max_neighbors=10):

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)

            with open(input_path, 'r') as f:
                data = json.load(f)
                bonds = data.get('bonds', [])
                contacts = data.get('contacts', [])

            neighbor_map = createSortedNeighbors(contacts, bonds, max_neighbors)

            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump({"neighbor_map": neighbor_map}, f, indent=4)


                
                



