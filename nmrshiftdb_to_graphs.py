import numpy as np
import pandas as pd
import pickle
import rdkit
from rdkit import Chem

from graph import Graph

# d_name = "nmrshiftdb.1H.pickle"
data = pickle.load(open('data/nmrshiftdb.1H.pickle', 'rb'))

BOND_TYPE_TO_LABEL = {
    rdkit.Chem.rdchem.BondType.SINGLE : 1,
    rdkit.Chem.rdchem.BondType.DOUBLE : 2,
    rdkit.Chem.rdchem.BondType.TRIPLE : 3,
    rdkit.Chem.rdchem.BondType.AROMATIC : 1.5
}

def load_graph(row):
    mol = Chem.Mol(row['rdmol'])
    n = mol.GetNumAtoms()
    edges, efs = [], []
    for e in mol.GetBonds():
        edges += [(e.GetBeginAtomIdx()+1, e.GetEndAtomIdx()+1)]
        efs += [BOND_TYPE_TO_LABEL[e.GetBondType()]]
    vfs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    labels = []
    sd = row['spect_dict'][0]
    for i in range(n):
        if i in sd:
            labels += [sd[i]]
        else:
            labels += [-100]
    return Graph(n, edges, vfs, efs), labels

d_train = {}
d_test = {}


for i, row in data.iterrows():
    g, l = load_graph(row)
    if row['morgan4_crc32']%10 <= 1:
        d_test[i] = (g, l)
    else:
        d_train[i] = (g, l)


pickle.dump(d_train, open('data/nmrshiftdb.1H.train.pickle', 'wb'))
pickle.dump(d_test, open('data/nmrshiftdb.1H.test.pickle', 'wb'))