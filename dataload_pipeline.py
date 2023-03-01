import numpy as np
import pandas as pd
import pickle

import os

import rdkit
from ruffus import *
from tqdm import tqdm

CSD_subsets = {
    'CSD-2k': {
        'filename': 'data/CSD-2k.txt'
    },

    'CSD-500': {
        'filename': 'data/CSD-500.txt'
    }
}

OUTPUT_DIR = "distance_features/" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params():
    for exp_name, exp_config in CSD_subsets.items():
        infiles = exp_config['file']
        outfiles = td(f"{exp_name}.pickle")
        yield infiles, outfiles, exp_name, exp_config

@mkdir(OUTPUT_DIR)
@files(params)
def convert_to_mols(infiles, outfiles, exp_name, exp_config):
    xyz_file = infiles
    with open(xyz_file) as file:
        xyz_lines = file.readLines()
    print('First line of file is', xyz_lines[0])

if __name__ == "__main__":
    pipeline_run([convert_to_mols], checksum_level=0)
