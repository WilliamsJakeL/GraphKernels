import numpy as np
import pandas as pd
import pickle

import os

from ruffus import *
from tqdm import tqdm

from graph import Graph

CSD_subsets = {
    'CSD-2k': {
        'filename': 'data/CSD-2k.txt'
    },

    'CSD-500': {
        'filename': 'data/CSD-500.txt'
    }
}

PROTEINS = {
    'PROTEINS': {
        'filename': 'data/PROTEINS.txt'
    } 
}

OUTPUT_DIR = "data/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params_CSD():
    for exp_name, exp_config in CSD_subsets.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.pickle")
        yield infiles, outfiles, exp_name, exp_config

def params_PROTEINS():
    for exp_name, exp_config in PROTEINS.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.pickle")
        yield infiles, outfiles, exp_name, exp_config

# @mkdir(OUTPUT_DIR)
# @files(params)
# def convert_CSDto_mols(infiles, outfiles, exp_name, exp_config):
#     xyz_file = infiles
#     with open(xyz_file) as file:
#         xyz_str = file.read()
#     xyz_lines = xyz_str.split('\n')
#     print('First line of file is', xyz_lines[0])
#     n = int(xyz_lines[0])
#     b1 = xyz_str[4:]
#     m = Chem.rdmolfiles.MolFromXYZBlock(b1)

@mkdir(OUTPUT_DIR)
@files(params_PROTEINS)
def convert_to_graphs(infiles, outfiles, exp_name, exp_config):
    print("Converting text to graphs:", exp_name)
    protein_file = infiles
    with open(protein_file) as file:
        lines = file.readlines()
    i = 1
    id = 0
    d = {}
    while i < len(lines):
        n, gf = lines[i].split()
        n, gf = int(n), int(gf)
        edges = []
        nfs = []
        for j in range(n):
            z = lines[i+j+1].split()
            nfs += [int(z[0])]
            for e2 in z[2:]:
                if int(e2) > j:
                    edges += [(j+1, int(e2)+1)]
        g = Graph(n, edges, nfs, [0 for _ in edges])
        d[id] = (g, gf)
        id += 1
        i += n + 1
    print("Found", len(d), "graphs.")
    pickle.dump(d, open(outfiles, 'wb'))


if __name__ == "__main__":
    pipeline_run([convert_to_graphs], checksum_level=0)

# with io.MoleculeReader(['ABEBUF', 'HXACAN', 'VUSDIX04']) as reader:
# ...     for mol in reader:
# ...         print(mol.identifier)
