import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from kernels import BhattKernel
from graph import Graph

g1 = Graph.generateRandom(N=10, p=0.25)
g2 = Graph.generateRandom(N=15, p=0.25)
g3 = Graph.generateRandom(N=10, p=0.5)
g4 = Graph.generateRandom(N=15, p=0.5)
g5 = Graph.generateRandom(N=10, p=0.75)
g6 = Graph.generateRandom(N=15, p=0.75)

K1 = BhattKernel(1, 20)
K2 = BhattKernel(0.05, 20)

a1 = K1.graphBinned(g1)
a2 = K1.graphBinned(g2)
a3 = K1.graphBinned(g3)
a4 = K1.graphBinned(g4)
a5 = K1.graphBinned(g5)
a6 = K1.graphBinned(g6)

b1 = K2.graphBinned(g1)
b2 = K2.graphBinned(g2)
b3 = K2.graphBinned(g3)
b4 = K2.graphBinned(g4)
b5 = K2.graphBinned(g5)
b6 = K2.graphBinned(g6)

fig, axs = plt.subplots(3,6,figsize=(12,6))

axs[0,0].imshow(g1.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[0,1].bar(list(range(20)), a1)
axs[0,2].bar(list(range(20)), b1)

axs[0,3].imshow(g2.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[0,4].bar(list(range(20)), a2)
axs[0,5].bar(list(range(20)), b2)

axs[1,0].imshow(g3.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[1,1].bar(list(range(20)), a3)
axs[1,2].bar(list(range(20)), b3)

axs[1,3].imshow(g4.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[1,4].bar(list(range(20)), a4)
axs[1,5].bar(list(range(20)), b4)

axs[2,0].imshow(g5.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[2,1].bar(list(range(20)), a5)
axs[2,2].bar(list(range(20)), b5)

axs[2,3].imshow(g6.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[2,4].bar(list(range(20)), a6)
axs[2,5].bar(list(range(20)), b6)

plt.show()