import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from kernels import BhattKernel
from graph import Graph

train_graphs = [Graph.generateRandom(N=10, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.75) for _ in range(100)]
train_y = [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)]
test_graphs = [Graph.generateRandom(N=10, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=17, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=17, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=17, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.75) for _ in range(100)]
test_y = [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)]
test_y_np = np.array(test_y)

# K = BhattKernel(t=[0.5], num_bins=40, graphs=train_graphs, y=train_y,r_lambda=100,calcWeights=True)

# preds = np.array([K.predictGraph(g) for g in test_graphs])

# print("Average absolute error:", np.mean(np.abs(preds - test_y_np)))
# print("Average absolute error (N=10):", np.mean(np.abs(preds[:300] - test_y_np[:300])))
# print("Average absolute error (N=25):", np.mean(np.abs(preds[300:600] - test_y_np[300:600])))
# print("Average absolute error (N=17):", np.mean(np.abs(preds[600:900] - test_y_np[600:900])))
# print("Average absolute error (N=50):", np.mean(np.abs(preds[900:] - test_y_np[900:])), '\n')
# print("Average absolute percent error:", np.mean(np.abs(preds-test_y_np)/test_y_np))

train_graphs_50 = [Graph.generateRandom(N=10, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=10, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=25, p=0.75) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.25) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.5) for _ in range(100)] + \
         [Graph.generateRandom(N=50, p=0.75) for _ in range(100)]
train_y_50 = [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)] + \
          [.25 for _ in range(100)] + \
          [.5 for _ in range(100)] + \
          [.75 for _ in range(100)]

print("\n Add in N=50 graphs! \n")

K = BhattKernel(t=[0.01, 0.1, 0.5, 1.0], num_bins=40, graphs=train_graphs_50, y=train_y_50,r_lambda=100,calcWeights=True)

preds = np.array([K.predictGraph(g) for g in test_graphs])

print("(50) Average absolute error:", np.mean(np.abs(preds - test_y_np)))
print("(50) Average absolute error (N=10):", np.mean(np.abs(preds[:300] - test_y_np[:300])))
print("(50) Average absolute error (N=25):", np.mean(np.abs(preds[300:600] - test_y_np[300:600])))
print("(50) Average absolute error (N=17):", np.mean(np.abs(preds[600:900] - test_y_np[600:900])))
print("(50) Average absolute error (N=50):", np.mean(np.abs(preds[900:] - test_y_np[900:])), '\n')
# print("Average absolute percent error:", np.mean(np.abs(preds-test_y_np)/test_y_np))

print("Test Full Random Graph Generation")

r = np.random.random((1000,2))
train_all_random = [Graph.generateRandom(int(n*98)+2, m) for (n,m) in r]
train_all_ry = [m for (_, m) in r]

K = BhattKernel(t=[0.01, 0.1, 0.5, 1.0], num_bins=40, graphs=train_all_random, y=train_all_ry,r_lambda=100,calcWeights=True)

preds = np.array([K.predictGraph(g) for g in test_graphs])

print("(50) Average absolute error:", np.mean(np.abs(preds - test_y_np)))
print("(50) Average absolute error (N=10):", np.mean(np.abs(preds[:300] - test_y_np[:300])))
print("(50) Average absolute error (N=25):", np.mean(np.abs(preds[300:600] - test_y_np[300:600])))
print("(50) Average absolute error (N=17):", np.mean(np.abs(preds[600:900] - test_y_np[600:900])))
print("(50) Average absolute error (N=50):", np.mean(np.abs(preds[900:] - test_y_np[900:])), '\n')
# print("Average absolute percent error:", np.mean(np.abs(preds-test_y_np)/test_y_np))