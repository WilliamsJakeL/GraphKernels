import numpy as np
import pandas as pd
import pickle

import os

from ruffus import *
from tqdm import tqdm

from graph import Graph

import rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolfiles import MolFromXYZBlock as xyzB

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

ATOM_TO_LABEL = {
    'H': 0,
    'C': 1,
    'O': 2,
    'N': 3
}

OUTPUT_DIR = "data/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params_CSD():
    for exp_name, exp_config in CSD_subsets.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.pickle")
        yield infiles, outfiles, exp_name, exp_config

def params_CSD_rdkit():
    for exp_name, exp_config in CSD_subsets.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.rdkit.pickle")
        yield infiles, outfiles, exp_name, exp_config

def params_PROTEINS():
    for exp_name, exp_config in PROTEINS.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.pickle")
        yield infiles, outfiles, exp_name, exp_config

BOND_TYPE_TO_LABEL = {
    rdkit.Chem.rdchem.BondType.SINGLE : 1,
    rdkit.Chem.rdchem.BondType.DOUBLE : 2,
    rdkit.Chem.rdchem.BondType.TRIPLE : 3,
    rdkit.Chem.rdchem.BondType.AROMATIC : 1.5
}

def getDistance(x1, y1, z1, x2, y2, z2):
    return np.linalg.norm(np.array([float(x1), float(y1), float(z1)]) - np.array([float(x2), float(y2), float(z2)]))

def load_graph(mol):
    # mol = Chem.Mol(row['rdmol'])
    n = mol.GetNumAtoms()
    edges, efs = [], []
    for e in mol.GetBonds():
        edges += [(e.GetBeginAtomIdx()+1, e.GetEndAtomIdx()+1)]
        efs += [BOND_TYPE_TO_LABEL[e.GetBondType()]]
    vfs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return Graph(n, edges, vfs, efs)

@mkdir(OUTPUT_DIR)
@files(params_CSD_rdkit)
def convert_CSD_to_rdkit_graphs(infiles, outfiles, exp_name, exp_config):
    print("Converting text to graphs:", exp_name)
    xyz_file = infiles
    with open(xyz_file) as file:
        lines = file.readlines()
    i = 0
    id = 0
    d = {}
    while i < len(lines):
        n = int(lines[i])
        shifts = []
        block = ''
        block += lines[i]
        block += lines[i+1]
        for j in range(n):
            z = lines[i+j+2].split()
            block += z[0] + '\t' + z[1] + '\t' + z[2] + '\t' + z[3] + '\n'
            shifts += [float(z[4])]
        m = xyzB(block)
        rdDetermineBonds.DetermineConnectivity(m)
        g = load_graph(m)
        d[id] = (g, shifts)
        id += 1
        i += n + 2
    print("Found", len(d), "graphs.")
    pickle.dump(d, open(outfiles, 'wb'))

@mkdir(OUTPUT_DIR)
@files(params_CSD)
def convert_CSD_to_graphs(infiles, outfiles, exp_name, exp_config):
    print("Converting text to graphs:", exp_name)
    xyz_file = infiles
    with open(xyz_file) as file:
        lines = file.readlines()
    i = 0
    id = 0
    d = {}
    while i < len(lines):
        n = int(lines[i])
        edges = []
        edge_feats = []
        nfs = []
        shifts = []
        for j in range(n):
            z = lines[i+j+2].split()
            nfs += [ATOM_TO_LABEL[z[0]]]
            x1 = float(z[1])
            y1 = float(z[2])
            z1 = float(z[3])
            shifts += [float(z[4])]
            for k in range(j+1, n):
                ze = lines[i+k+2].split()
                x2 = float(ze[1])
                y2 = float(ze[2])
                z2 = float(ze[3])
                edges += [(j+1, k+1)]
                edge_feats += [1/getDistance(x1, y1, z1, x2, y2, z2)]
        g = Graph(n, edges, nfs, edge_feats)
        d[id] = (g, shifts)
        id += 1
        i += n + 2
    print("Found", len(d), "graphs.")
    pickle.dump(d, open(outfiles, 'wb'))

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
        g = Graph(n, edges, nfs, [1 for _ in edges])
        d[id] = (g, gf)
        id += 1
        i += n + 1
    print("Found", len(d), "graphs.")
    pickle.dump(d, open(outfiles, 'wb'))

if __name__ == "__main__":
    pipeline_run([convert_CSD_to_rdkit_graphs, convert_CSD_to_graphs, convert_to_graphs], checksum_level=0)
