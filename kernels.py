from re import S
import numpy as np
from scipy.linalg import expm
from graph import Graph

class BhattKernel:

    def __init__(self, t, num_bins, graphs=None, y=None, r_lambda=0, use_labels=False, label_pairs=None, calcWeights=False):
        self.t = t
        self.numT = len(t)
        self.num_bins = num_bins
        self.graphs = graphs
        self.y = y
        self.r_lambda = r_lambda
        self.use_labels = use_labels
        if self.use_labels and label_pairs is None:
            raise Exception("No label pairs provided")
        self.label_pairs = label_pairs
        if not self.graphs is None:
            if self.numT == 1:
                self.binnedGraphs = [self.graphBinnedSingle(g) for g in self.graphs]
            else:
                self.binnedGraphs = [self.graphBinned(g) for g in self.graphs]
        if not calcWeights:
            self.weights = np.zeros(self.num_bins)
        else:
            self.weights, err = self.calculateWeights()
            print("Train Set Average Absolute Error:", err)
        
    def predictGraph(self, g):
        if self.numT == 1:
            return self.weights@np.sqrt(self.graphBinnedSingle(g))
        else:
            return self.weights@np.sqrt(self.graphBinned(g))

    def graphBinnedSingle(self, g):
        L = g.getLaplacian()
        h = expm(-self.t[0]*L)
        if self.use_labels:
            bins = {}
            for p in self.label_pairs:
                bins[p] = [0 for _ in range(self.num_bins)]
            it = np.nditer(h, flags=['multi_index'])
            while not it.finished:
                v = it[0]
                x, y = g.vert_feats[min(it.multi_index)], g.vert_feats[max(it.multi_index)]
                x, y = min(x,y), max(x,y)
                if v == 1:
                    bins[(x,y)][self.num_bins-1] += 1
                else:
                    bins[(x,y)][int(v*self.num_bins)] += 1
                it.iternext()
            pi = []
            for l in bins.values():
                pi += l
            return pi
        else:
            pi = [0 for _ in range(self.num_bins)]
            for v in np.nditer(h):
                if v == 1:
                    pi[self.num_bins-1] += 1
                else:
                    pi[int(v*self.num_bins)] += 1
            return pi 

    def graphBinned(self, g):
        L = g.getLaplacian()
        # h = expm(-(self.t/g.vertN)*L)
        w, v = np.linalg.eigh(L)
        pi_t = []
        for i, B in enumerate(self.t):
            h = v@np.diag(np.exp(-B*w))@v.T
            if self.use_labels:
                bins = {}
                for p in self.label_pairs:
                    bins[p] = [0 for _ in range(self.num_bins)]
                it = np.nditer(h, flags=['multi_index'])
                while not it.finished:
                    entry = it[0]
                    x, y = g.vert_feats[min(it.multi_index)], g.vert_feats[max(it.multi_index)]
                    x, y = min(x,y), max(x,y)
                    if entry == 1:
                        bins[(x,y)][self.num_bins-1] += 1
                    else:
                        bins[(x,y)][int(entry*self.num_bins)] += 1
                    it.iternext()
                pi = []
                for l in bins.values():
                    pi += l
                pi_t += pi
            else:
                pi = [0 for _ in range(self.num_bins)]
                for entry in np.nditer(h):
                    if entry == 1:
                        pi[self.num_bins-1] += 1
                    else:
                        pi[int(entry*self.num_bins)] += 1
                pi_t += pi 
        return pi_t

    def calculateWeights(self):
        phi = np.sqrt(self.binnedGraphs)
        alpha = np.linalg.solve((phi@phi.T) + self.r_lambda*np.eye(len(self.graphs)), self.y)
        w = phi.T@alpha
        return w, np.mean(np.abs(phi@w - self.y))

class BhattKernelNodes:

    def __init__(self, t, num_bins, graphs=None, ys=None, r_lambda=0, use_labels=False, label_types=None, calcWeights=False):
        self.t = t
        self.numT = len(t)
        self.num_bins = num_bins
        self.graphs = graphs
        self.ys = ys
        self.r_lambda = r_lambda
        self.use_labels = use_labels
        if self.use_labels and label_types is None:
            raise Exception("No label pairs provided")
        self.label_pairs = label_types
        self.binnedGraphs = []
        self.y = []
        if not self.graphs is None:
            if self.numT == 1:
                for g, (node, y) in zip(self.graphs, self.ys):
                    self.binnedGraphs += self.graphBinnedSingle(g, node)
                    self.y += [y]
            else:
                for g, (node, y) in zip(self.graphs, self.ys):
                    self.binnedGraphs += self.graphBinned(g, node)
                    self.y += [y]
        if not calcWeights:
            self.weights = np.zeros(self.num_bins)
        else:
            self.weights, err = self.calculateWeights()
            print("Train Set Average Absolute Error:", err)
        
    def predictNode(self, g, node):
        if self.numT == 1:
            return self.weights@np.sqrt(self.graphBinnedSingle(g, node))
        else:
            return self.weights@np.sqrt(self.graphBinned(g, node))

    def graphBinnedSingle(self, g, node):
        L = g.getLaplacian()
        h = expm(-self.t[0]*L)
        if self.use_labels:
            bins = {}
            for p in self.label_types:
                bins[p] = [0 for _ in range(self.num_bins)]
            for it, v in enumerate(h[node]):
                x = g.vert_feats[it]
                if v == 1:
                    bins[x][self.num_bins-1] += 1
                else:
                    bins[x][int(v*self.num_bins)] += 1
            pi = []
            for l in bins.values():
                pi += l
            return pi
        else:
            pi = [0 for _ in range(self.num_bins)]
            for v in h[node]:
                if v == 1:
                    pi[self.num_bins-1] += 1
                else:
                    pi[int(v*self.num_bins)] += 1
            return pi 

    def graphBinned(self, g, node):
        L = g.getLaplacian()
        # h = expm(-(self.t/g.vertN)*L)
        w, v = np.linalg.eigh(L)
        pi_t = []
        for i, B in enumerate(self.t):
            h = v@np.diag(np.exp(-B*w))@v.T
            if self.use_labels:
                bins = {}
                for p in self.label_types:
                    bins[p] = [0 for _ in range(self.num_bins)]
                for it, entry in enumerate(h[node]):
                    x = g.vert_feats[it]
                    if entry == 1:
                        bins[x][self.num_bins-1] += 1
                    else:
                        bins[x][int(entry*self.num_bins)] += 1
                pi = []
                for l in bins.values():
                    pi += l
                pi_t += pi
            else:
                pi = [0 for _ in range(self.num_bins)]
                for entry in h[node]:
                    if entry == 1:
                        pi[self.num_bins-1] += 1
                    else:
                        pi[int(entry*self.num_bins)] += 1
                pi_t += pi 
        return pi_t

    def calculateWeights(self):
        phi = np.sqrt(self.binnedGraphs)
        alpha = np.linalg.solve((phi@phi.T) + self.r_lambda*np.eye(len(self.graphs)), self.y)
        w = phi.T@alpha
        return w, np.mean(np.abs(phi@w - self.y))