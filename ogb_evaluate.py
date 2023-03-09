from kernels import BhattKernel, BhattKernelNodes
from graph import Graph

import numpy as np
import pandas as pd
import pickle
from scipy.stats import kendalltau as kt

import os

# from ogb.graphproppred import Evaluator

d_name = "ogbg-molfreesolv"

exp_config = {
    't': [0.001, 0.01, 0.1, 1, 10],
    'num_bins': 40,
    'r_lambda': 100,
    'label_pairs': [(5,5), (5,6), (5,7), (5,8), (5,14), (5, 15), (5,16), (5, 34), (5, 52),
                       (6, 6), (6,7), (6,8), (6,14), (6, 15), (6,16), (6, 34), (6, 52),
                       (7, 7), (7,8), (7,14), (7, 15), (7,16), (7, 34), (7, 52),
                       (8, 8), (8,14), (8, 15), (8,16), (8, 34), (8, 52),
                       (14, 14), (14, 15), (14,16), (14, 34), (14, 52),
                       (15, 15), (15,16), (15, 34), (15, 52),
                       (16, 16), (16, 34), (16, 52),
                       (34, 34), (34, 52), (52, 52)]
}

# exp_config = {
#     't': [0.001, 0.01, 0.1, 1, 10],
#     'num_bins': 40,
#     'r_lambda': 100,
#     'label_pairs': [(4,4), (4,5), (4,6), (4,7), (4,8), (4, 13), (4,14), (4, 15), (4,16), (4, 34), (4, 52),
#                        (5,5), (5,6), (5,7), (5,8), (5, 13), (5,14), (5, 15), (5,16), (5, 34), (5, 52),
#                        (6, 6), (6,7), (6,8), (6, 13), (6,14), (6, 15), (6,16), (6, 34), (6, 52),
#                        (7, 7), (7,8), (7, 13), (7,14), (7, 15), (7,16), (7, 34), (7, 52),
#                        (8, 8), (8, 13), (8,14), (8, 15), (8,16), (8, 34), (8, 52),
#                        (13, 13), (13, 14), (13, 15), (13,16), (13, 34), (13, 52),
#                        (14, 14), (14, 15), (14,16), (14, 34), (14, 52),
#                        (15, 15), (15,16), (15, 34), (15, 52),
#                        (16, 16), (16, 34), (16, 52),
#                        (34, 34), (34, 52), (52, 52)]
# }

pl = exp_config.get('label_pairs', None)

print("Loading Molecules")
d = pickle.load(open('data/OGB_molfreesolv_1.train.pickle', 'rb'))
train_graphs = [g for g, _ in d.values()]
y = [l[0] for _, l in d.values()]

print("Training Kernels")
kernel = BhattKernel(exp_config['t'], exp_config['num_bins'], train_graphs, y, exp_config['r_lambda'], True, pl, calcWeights=True)
kernel_nl = BhattKernel(exp_config['t'], exp_config['num_bins'], train_graphs, y, exp_config['r_lambda'], False, pl, calcWeights=True)


print("Loading Test")
d_test = pickle.load(open('data/OGB_molfreesolv_1.test.pickle', 'rb'))
test_graphs = [g for g, _ in d_test.values()]
y_test = [l[0] for _, l in d_test.values()]

preds = [kernel.predictGraph(g) for g in test_graphs]
preds_nl = [kernel_nl.predictGraph(g) for g in test_graphs]

preds_d = {'Average Error (MAE)': np.mean(np.abs(np.array(preds) - np.array(y_test))),
                'RMSE': np.sqrt(np.mean((np.array(preds) - np.array(y_test))**2)),
                'Kendall': kt(preds, y_test)}
preds_d_nl = {'Average Error (MAE)': np.mean(np.abs(np.array(preds_nl) - np.array(y_test))),
                'RMSE': np.sqrt(np.mean((np.array(preds_nl) - np.array(y_test))**2)),
                'Kendall': kt(preds_nl, y_test)}
with open('predictions/OGB_molfreesolv_1.txt', 'w') as file:
    file.write("Labels: ")
    file.write(str(preds_d))
    file.write('\nNo Labels: ')
    file.write(str(preds_d_nl))