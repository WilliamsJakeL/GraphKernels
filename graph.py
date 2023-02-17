import numpy as np
from queue import PriorityQueue

class Graph:

    def __init__(self, num_verts, edges):
        self.vertN = num_verts
        self.validateEdges(edges)
        self.edges = edges

    def validateEdges(self, edges):
        for e1, e2 in edges:
            if not self.isValidEdge(e1, e2):
                raise ValueError("Invalid edge endpoints provided: " + str(e1) + ", " + str(e2) + ".")

    def isValidEdge(self, e1, e2):
        return e1 in range(1, self.vertN+1) and e2 in range(1, self.vertN+1)

    def addEdge(self, e1, e2):
        self.validateEdges([(e1, e2)])
        self.edges += [(e1, e2)]

    def getDegrees(self):
        d = [0 for _ in range(self.vertN)]
        for e1, e2 in self.edges:
            d[e1-1] += 1
            d[e2-1] += 1
        return d
    
    def getAdjacencyMatrix(self):
        a = np.zeros((self.vertN, self.vertN))
        for e1, e2 in self.edges:
            a[e1-1][e2-1] = 1
            a[e2-1][e1-1] = 1
        return a

    def getLaplacian(self):
        return np.diag(self.getDegrees()) - self.getAdjacencyMatrix()

    def generateRandom(N=10, p=0.5):
        e = []
        r = np.random.random((N*(N-1))//2)
        c = 0
        for i in range(1,N+1):
            for j in range(i+1, N+1):
                if r[c] > p:
                    e += [(i,j)]
                c += 1
        return Graph(N, e)

    def hasSixRing(self):
        for i in self.vertN:
            return True
