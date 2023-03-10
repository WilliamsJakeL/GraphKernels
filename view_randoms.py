import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from kernels import BhattKernel
from graph import Graph

g1 = Graph.generateRandom(N=10, p=0.25)
g2 = Graph.generateRandom(N=25, p=0.25)
g3 = Graph.generateRandom(N=10, p=0.5)
g4 = Graph.generateRandom(N=25, p=0.5)
g5 = Graph.generateRandom(N=10, p=0.75)
g6 = Graph.generateRandom(N=25, p=0.75)

K1 = BhattKernel([0.001,0.01, 0.1, 1, 10], 40)

a1 = K1.graphBinned(g1)
a2 = K1.graphBinned(g2)
a3 = K1.graphBinned(g3)
a4 = K1.graphBinned(g4)
a5 = K1.graphBinned(g5)
a6 = K1.graphBinned(g6)

# b1 = K2.graphBinned(g1)
# b2 = K2.graphBinned(g2)
# b3 = K2.graphBinned(g3)
# b4 = K2.graphBinned(g4)
# b5 = K2.graphBinned(g5)
# b6 = K2.graphBinned(g6)

fig, axs = plt.subplots(6,6,figsize=(8,6))

axs[0,0].imshow(g1.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[0,1].bar(list(range(40)), a1[:40])
axs[0,2].bar(list(range(40)), a1[40:80])
axs[0,3].bar(list(range(40)), a1[80:120])
axs[0,4].bar(list(range(40)), a1[120:160])
axs[0,5].bar(list(range(40)), a1[160:200])
# axs[0,2].bar(list(range(20)), b1)

axs[1,0].imshow(g2.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[1,1].bar(list(range(40)), a2[:40])
axs[1,2].bar(list(range(40)), a2[40:80])
axs[1,3].bar(list(range(40)), a2[80:120])
axs[1,4].bar(list(range(40)), a2[120:160])
axs[1,5].bar(list(range(40)), a2[160:200])
# axs[0,5].bar(list(range(20)), b2)

axs[2,0].imshow(g3.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[2,1].bar(list(range(40)), a3[:40])
axs[2,2].bar(list(range(40)), a3[40:80])
axs[2,3].bar(list(range(40)), a3[80:120])
axs[2,4].bar(list(range(40)), a3[120:160])
axs[2,5].bar(list(range(40)), a3[160:200])
# axs[1,2].bar(list(range(20)), b3)

axs[3,0].imshow(g4.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[3,1].bar(list(range(40)), a4[:40])
axs[3,2].bar(list(range(40)), a4[40:80])
axs[3,3].bar(list(range(40)), a4[80:120])
axs[3,4].bar(list(range(40)), a4[120:160])
axs[3,5].bar(list(range(40)), a4[160:200])
# axs[1,5].bar(list(range(20)), b4)

axs[4,0].imshow(g5.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[4,1].bar(list(range(40)), a5[:40])
axs[4,2].bar(list(range(40)), a5[40:80])
axs[4,3].bar(list(range(40)), a5[80:120])
axs[4,4].bar(list(range(40)), a5[120:160])
axs[4,5].bar(list(range(40)), a5[160:200])
# axs[2,2].bar(list(range(20)), b5)

axs[5,0].imshow(g6.getAdjacencyMatrix(), cmap='hot', interpolation='nearest')
axs[5,1].bar(list(range(40)), a6[:40])
axs[5,2].bar(list(range(40)), a6[40:80])
axs[5,3].bar(list(range(40)), a6[80:120])
axs[5,4].bar(list(range(40)), a6[120:160])
axs[5,5].bar(list(range(40)), a6[160:200])
# axs[2,5].bar(list(range(20)), b6)

for z in axs:
    for ax in z:
        ax.set_xticks([]) 
        ax.set_yticks([])

rows = ['N=10,\np=0.25', 'N=25,\np=0.25', 'N=10,\np=0.5', 'N=25,\np=0.5', 'N=10,\np=0.75', 'N=25,\np=0.75']
cols = ['', 't=0.001', 't=0.01', 't=0.1', 't=1', 't=10']

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

pad = 5
for ax, row in zip(axs[:,0], rows):
    # ax.set_ylabel(row, rotation=0, size='large')
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.tight_layout()
plt.show()