from kernels import BhattKernel
import numpy as np
import pandas as pd
import pickle

import os
from collections import Counter

from ruffus import *
from tqdm import tqdm

from graph import Graph

EXPERIMENTS = {
    'PROTEINS_nolabels': {
        'filename': 'data/PROTEINS.pickle',
        'train_split_seed': 0,
        't': [0.001, 0.01, 0.1, 1, 10],
        'num_bins': 40,
        'r_lambda': 100,
        'use_labels': False
    },

    'PROTEINS_labels': {
        'filename': 'data/PROTEINS.pickle',
        'train_split_seed': 0,
        't': [0.001, 0.01, 0.1, 1, 10],
        'num_bins': 40,
        'r_lambda': 100,
        'use_labels': True,
        'label_pairs': [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    }
}

OUTPUT_DIR = "predictions/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params():
    for exp_name, exp_config in EXPERIMENTS.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.knn.pickle"), td(f"{exp_name}.preds.txt")
        yield infiles, outfiles, exp_name, exp_config

@mkdir(OUTPUT_DIR)
@files(params)
def train_KNN(infiles, outfiles, exp_name, exp_config):
    graph_file = infiles
    kernel_file, preds_files = outfiles
    print("Loading Graphs to train KNN")
    graphs = pickle.load(open(graph_file, 'rb'))
    rng = np.random.default_rng(seed=exp_config['train_split_seed'])
    N = len(graphs)
    indices = np.arange(N)
    rng.shuffle(indices)
    train_graphs = []
    y = []
    c = Counter()
    for i in indices[:int(N*.8)]:
        train_graphs += [graphs[i][0]]
        y += [graphs[i][1]]
        c[graphs[i][1]] += 1
    print("Training KNN")
    pl = exp_config.get('label_pairs', None)
    kernel = BhattKernel(exp_config['t'], exp_config['num_bins'], train_graphs, y, exp_config['r_lambda'], exp_config['use_labels'], pl, calcWeights=True)
    pickle.dump(kernel, open(kernel_file, 'wb'))
    print("Loading Graphs for Testing KNN")
    test_graphs = []
    y_test = []
    c_test = Counter()
    for i in indices[int(N*.8):]:
        test_graphs += [graphs[i][0]]
        y_test += [graphs[i][1]]
        c_test[graphs[i][1]] += 1
    print("Testing KNN")
    preds = [kernel.predictGraph(g) for g in test_graphs]
    correct = 0
    for p, l in zip(preds, y_test):
        correct += 1 - abs(l - round(p))
    preds_d = {'Label Counts (Train)': str(c),
                'Label Counts (Test)': str(c_test),
                'Average Error': np.mean(np.abs(np.array(preds) - np.array(y_test))),
                'Accuracy': correct/len(y_test)}
    with open(preds_files, 'w') as file:
        file.write(str(preds_d))

if __name__ == "__main__":
    pipeline_run([train_KNN], checksum_level=0)