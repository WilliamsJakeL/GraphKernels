import numpy as np
import pandas as pd
import pickle

import os

from ruffus import *
from tqdm import tqdm

from graph import Graph

EXPERIMENTS = {
    'PROTEINS_basic': {
        'filename': 'data/PROTEINS.pickle'
    }
}

OUTPUT_DIR = "predictions/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params():
    for exp_name, exp_config in EXPERIMENTS.items():
        infiles = exp_config['filename']
        outfiles = td(f"{exp_name}.knn.pickle")
        yield infiles, outfiles, exp_name, exp_config

@mkdir(OUTPUT_DIR)
@files(params)
def train_KNN(infiles, outfiles, exp_name, exp_config):
    graph_file = infiles
    graphs = pickle.load(open(graph_file, 'rb'))
    print(len(graphs))
    pickle.dump('a', open(outfiles, 'wb'))

@transform(train_KNN, 
           suffix(".knn.pickle"), 
           ".eval.txt")
def evaluate(infiles, outfile):
    knn_file = infiles
    knn = pickle.load(open(knn_file, 'rb'))
    print(knn)
    pickle.dump('b', open(outfile, 'wb'))

if __name__ == "__main__":
    pipeline_run([train_KNN, evaluate], checksum_level=0)