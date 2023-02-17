import numpy as np
from scipy.linalg import expm
from graph import Graph

class BhattKernel:

    def __init__(self, t, num_bins, graphs=None, y=None, r_lambda=0, calcWeights=False):
        self.t = t
        self.num_bins = num_bins
        self.graphs = graphs
        self.y = y
        self.r_lambda = r_lambda
        if not self.graphs is None:
            self.binnedGraphs = [self.graphBinned(g) for g in self.graphs]
        if not calcWeights:
            self.weights = np.zeros(self.num_bins)
        else:
            self.weights, err = self.calculateWeights()
            print("Train Set Average Absolute Error:", err)
        
    def predictGraph(self, g):
        return self.weights@np.sqrt(self.graphBinned(g))

    def graphBinned(self, g):
        L = g.getLaplacian()
        h = expm(-(self.t/g.vertN)*L)
        pi = [0 for _ in range(self.num_bins)]
        for v in np.nditer(h):
            if v == 1:
                pi[self.num_bins-1] += 1
            else:
                pi[int(v*self.num_bins)] += 1
        return pi

    def calculateWeights(self):
        phi = np.sqrt(self.binnedGraphs)
        alpha = np.linalg.solve((phi@phi.T) + self.r_lambda*np.eye(len(self.graphs)), self.y)
        w = phi.T@alpha
        return w, np.mean(np.abs(phi@w - self.y))