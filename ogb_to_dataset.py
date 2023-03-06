import numpy as np
import pickle

from graph import Graph

from ogb.graphproppred import GraphPropPredDataset

d_name = "ogbg-molesol"

dataset = GraphPropPredDataset(name = d_name)

def load_graph(g):
    n = g['num_nodes']
    edges, efs = [], []
    for i, (e1, e2) in enumerate(g['edge_index'].T):
        if e1 < e2:
            edges += [(e1+1, e2+1)]
            efs += [g['edge_feat'][i][0]]
    vfs = [a[0] for a in g['node_feat']]
    return Graph(n, edges, vfs, efs)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

train_d = {}
for i in train_idx:
    graph, label = dataset[i]
    train_d[i] = (load_graph(graph), label)
test_d = {}
for i in test_idx:
    graph, label = dataset[i]
    test_d[i] = (load_graph(graph), label)
valid_d = {}
for i in valid_idx:
    graph, label = dataset[i]
    valid_d[i] = (load_graph(graph), label)

pickle.dump(train_d, open('data/OGB_molesol.train.pickle', 'wb'))
pickle.dump(test_d, open('data/OGB_molesol.test.pickle', 'wb'))
pickle.dump(valid_d, open('data/OGB_molesol.valid.pickle', 'wb'))