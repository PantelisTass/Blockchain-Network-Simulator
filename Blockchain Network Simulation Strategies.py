#!/usr/bin/env python
# coding: utf-8

# In[1132]:


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])
            
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        Min = 1e7
        min_index = 0
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < Min and sptSet[v] == False:
                Min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    
    def dijkstra(self, src):
     
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
 
        #self.printSolution(dist)
        return dist
    
    def Eigenvector_Centrality(self):
        # normalize starting vector
        x = dict([(n,1.0/self.V) for n in range(self.V)])
        s = 1.0/sum(x.values())
        for k in x:
            x[k] *= s
        Number_Nodes = self.V

        # make up to max_iter iterations
        max_iter = 50
        for i in range(max_iter):
            xlast = x
            x = dict.fromkeys(xlast, 0)

            # do the multiplication y = Cx
            # C is the matrix with entries
            α = [xlast[k] for k in range(self.V)]
            C = [[0 for _ in range(self.V)] for _ in range(self.V)]
            for i in range(self.V):
                for j in range(self.V):
                    if self.graph[i][j] != 1E7:
                        C[i][j] = 1
            B = np.matrix(C).dot(α)
            x = dict((n, B.item(n)) for n in range(self.V))

            # normalize vector
            try:
                s = 1.0/sqrt(sum(v**2 for v in x.values()))

            # this should never be zero?
            except ZeroDivisionError:
                s = 1.0
            for n in x:
                x[n] *= s

            # check convergence
            tol = 1E-5
            err = sum([abs(x[n]-xlast[n]) for n in x])
            if err < Number_Nodes*tol:
                return x
        return x
  
class Node:
    import time
    def __init__(self, dataval=None, Time=time.time()):
        self.dataval = dataval
        self.time = Time
        self.nextval = None

class time_series():
    def __init__(self):
        self.headval = None
    


# In[1347]:


import numpy as np
from scipy.stats import skewnorm
Regions_Distribution = {"NORTH_AMERICA": 0.3316, "EUROPE": 0.4998, "SOUTH_AMERICA":0.0090, "ASIA_PACIFIC":0.1177
                        , "JAPAN":0.0224,
                    "AUSTRALIA":0.0195}

Regions_Bandwidth = { "NORTH_AMERICA":19200000, "EUROPE": 20700000, "SOUTH_AMERICA": 5800000, "ASIA_PACIFIC":15700000
                        , "JAPAN": 10200000,
                    "AUSTRALIA":11300000}

Region_Latency = [[32, 124, 184, 198, 151, 189],
      [124, 11, 227, 237, 252, 294],
      [184, 227, 88, 325, 301, 322],
      [198, 237, 325, 85, 58, 198],
      [151, 252, 301, 58, 12, 126],
      [189, 294, 322, 198, 126, 16]]

#Latency distribution reference:
'''Gencer, A. E., Basu, S., Eyal, I., Van Renesse, R. & Sirer, E. G. Decentralization in bitcoin and ethereum net-
works. In International Conference on Financial Cryptography and Data Security, 439–457 (Springer, 2018).'''
#Use heavy-tail skew distribution (skew-normal or weibul)



Nodes_Region = [33,50,1,12,2,2]
Number_Nodes = 100
Nodes_Cumulative = np.cumsum(Nodes_Region)
Intervals = [list(range(Nodes_Cumulative[0]))]+[list(range(Nodes_Cumulative[i], Nodes_Cumulative[i+1])) for i in range(5)]


def gamma_dist(N,M,C, Δτ, prevNetwork):
    Gamma = []
    for k in range(C):

        Blockchain = Graph(Number_Nodes)
        W = [[0 for _ in range(Number_Nodes)] for _ in range(Number_Nodes)]
        '''
        from math import comb
        K = comb(Number_Nodes,2)
        P = [np.random.uniform(-1,1) for _ in range(K)] 
        #M is the Correlation weight matrix - specific interpretation will be assigned later
        M = [[np.random.randint(1) for _ in range(K)] for _ in range(K)]
        #B = Bias , perhaps relate to bandwidth
        B = [np.random.uniform(0,1) for _ in range(K)]
        α = vecsigm(np.matrix(M).dot(P)+B)
        count = 0 
        '''
        #adjacency matrix 
        A = [[0 for _ in range(Number_Nodes)] for _ in range(Number_Nodes)]
        
        for i in range(Number_Nodes-1):
            for j in range(i+1, Number_Nodes):
                #previous network state influences connectivity
                if np.random.uniform(0,1) >= 0.1:
                    A[i][j] = 1
                    A[j][i] = 1
                                     
        
        Blockchain_unweighted = Graph(Number_Nodes)
        Blockchain_unweighted.graph = A

        #Eigenvector Centrality ranking of nodes - high score means node is connected to many highly connected nodes
        #The latency in the network is adjusted by the above centrality measure that accounts for topological properties of network
        
                
        Ω = Blockchain_unweighted.Eigenvector_Centrality()  
        
        for i in range(Number_Nodes-1):
            for j in range(i+1, Number_Nodes):
                for l in range(5):
                    for m in range(l,6):
                        if (i in Intervals[l]) and (j in Intervals[m]):
                            #previous network state influences connectivity
                            if prevNetwork.graph[i][j] != 1E7:
                                c = (Ω[i]+Ω[j])
                                if A[i][j] == 1:
                                    
                                    rand =  prevNetwork.graph[i][j] + prevNetwork.graph[i][j]*Δτ*skewnorm.rvs(3*c, size = None)
                                    W[i][j] = rand 
                                    W[j][i] = rand
                                    
                                else: 
                                    W[i][j] = 1E7
                                    W[j][i] = 1E7  
                            else:
                                c = (Ω[i]+Ω[j])
                               
                                rand = Region_Latency[l][m] + Region_Latency[l][m]*Δτ*skewnorm.rvs(3*c, size = None)
                                W[i][j] = rand
                                W[j][i] = rand
                                    


        Blockchain.graph = W
        dist_N = Blockchain.dijkstra(N)
        dist_M = Blockchain.dijkstra(M)
        N_close = 0
        gamma = 0
                
        for i in set(range(Number_Nodes))-{N,M}:
            #Effect of time interval on next iteration of network 
            #np.random.poisson(lam = Ω[M])*
            if dist_N[i] < dist_M[i]:
                N_close += 1
        gamma = N_close/Number_Nodes
        Gamma.append(gamma)
    return (Blockchain, Gamma)
    #plt.hist(Gamma, bins = 10, density = True)
    #plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))
vecsigm = np.vectorize(sigmoid)


def generate_blockchain(Number_Nodes):
    Blockchain = Graph(Number_Nodes)
    #adjacency matrix 
    W = [[0 for _ in range(Number_Nodes)] for _ in range(Number_Nodes)]
    '''
    from math import comb
    K = comb(Number_Nodes,2)
    P = [np.random.uniform(-1,1) for _ in range(K)] 
    #M is the Correlation weight matrix - specific interpretation will be assigned later
    M = [[np.random.randint(1) for _ in range(K)] for _ in range(K)]
    #B = Bias , perhaps relate to bandwidth
    B = [np.random.uniform(0,1) for _ in range(K)]
    α = vecsigm(np.matrix(M).dot(P)+B)
    count = 0 
    '''
    for i in range(Number_Nodes-1):

        for j in range(i+1, Number_Nodes):
            for l in range(5):
                for m in range(l,6):
                    if (i in Intervals[l]) and (j in Intervals[m]):

                        if np.random.uniform(0,1)<0.01:

                            #if connection is not active - SimBlock Paper implementation
                            W[i][j] = 1E7
                            W[j][i] = 1E7
                        else:
                            #latency if connection is active -  SimBlock Paper implementaiton
                            mean = Region_Latency[l][m]
                            rand = np.random.poisson(mean)
                            shape = 0.2 * mean;
                            scale = mean - 5;
                            rand = int(scale / pow(np.random.uniform(0,1), 1.0 / shape))
                            W[i][j] = rand
                            W[j][i] = rand
            #count+=1
    Blockchain.graph = W
    return Blockchain


# In[688]:


#QQ plot monte carlo simulation of gamma
import statistics as stats
import statsmodels.api as sm
import pylab as py
v = (np.subtract(a,[stats.mean(a)]*500)/(stats.stdev(a)))
sm.qqplot(v, line ='45')


py.show()
#stats.stdev(Gamma)


# In[137]:


#Region Graph Gamma 
Region_Latency = [[32, 124, 184, 198, 151, 189],
      [124, 11, 227, 237, 252, 294],
      [184, 227, 88, 325, 301, 322],
      [198, 237, 325, 85, 58, 198],
      [151, 252, 301, 58, 12, 126],
      [189, 294, 322, 198, 126, 16]]

Blockchain = Graph(6)
N=0
M=5
Blockchain.graph = Region_Latency
dist_N = Blockchain.dijkstra(N)
dist_M = Blockchain.dijkstra(M)

N_close = 0
gamma = 0

for i in set(range(6))-{N,M}:
    if dist_N[i] < dist_M[i]:
        N_close += 1
gamma = N_close/6
gamma


# In[388]:


def distribution(N,M):
    Gamma = []
    for k in range(1000):
        Blockchain = Graph(Card)
        W = [[0 for _ in range(Card)] for _ in range(Card)]
        for i in range(Card-1):
            for j in range(i+1, Card):
                rand_gauss = max(float(np.random.normal(loc=0.0, scale=1, size=1)),0)
                rand_unif = float(np.random.uniform(0,10))
                rand_poisson = int(np.random.poisson(lam = 1, size = 1 ))
                W[i][j] = rand_unif
                W[j][i] = rand_unif
        Blockchain.graph = W
        dist_N = Blockchain.dijkstra(N)
        dist_M = Blockchain.dijkstra(M)

        N_close = 0
        gamma = 0

        for i in set(range(Card))-{N,M}:
            if sigmoid(dist_N[i]) < sigmoid(dist_M[i]):
                N_close += 1
        gamma = N_close/Card
        Gamma.append(gamma)

    plt.hist(Gamma, bins = 10, density = True)
    plt.show()

    # This code is contributed by Divyanshu Mehta

distribution(0,1)


# In[ ]:


G = Graph(3)
G.graph = [[0,0,1],[10000,0,50],[1,50,0]]
Ω = G.Eigenvector_Centrality()
list(np.matmul(G.graph,[[n] for n in range(G.V)]))
for i in range(G.V):
    G.graph[i] = [G.graph[i][j]*Ω[i] for j in range(G.V)]
G.graph
Ω


# In[1008]:


def stopping_time_simulator(τ, N, M):
    # give time series of gamma between two nodes on the blockchain sampled at times τ
    gamma = []
    Number_Nodes = 100
    Network = time_series()
    network_init = generate_blockchain(Number_Nodes)
    Ω = network_init.Eigenvector_Centrality()        
    #bias parameter for activaiton in network:

    
    dist_N = network_init.dijkstra(N)
    dist_M = network_init.dijkstra(M)
    N_close = 0
    γ = 0

    for i in set(range(Number_Nodes))-{N,M}:
        #Effect of time interval on next iteration of network 
        if dist_N[i] < dist_M[i]:
            N_close += 1
    γ = N_close/Number_Nodes
    gamma += [γ]
    
    Network.headval = Node(network_init, τ[0])
    pointer = Network.headval
    
    for n in range(1,len(τ)):
        prevNetwork = pointer.dataval
        α = gamma_dist(N,M,1, τ[n]-τ[n-1], prevNetwork)
        pointer.nextval = Node(α[0], τ[n])
        pointer = pointer.nextval
        gamma += α[1]
    return (Network, gamma)


# In[1354]:


T = 1000
τ = [0]*T
for n in range(1,T):
    τ[n] = np.random.uniform(τ[n-1], τ[n-1]+.01)
τ2 = [n for n in range(T)]
y = stopping_time_simulator(τ, 0, 10)
Y = [sum(y[1][:n])/n for n in range(1, T+1)]


plt.plot(τ, y[1], linewidth = 0.8)
plt.plot(τ, Y)


# In[1328]:


Transitions = {0:[0,0.6], 1: [0.6001, 0.69999], 2: [0.7, 1]}
P = [[0 for _ in range(3)] for _ in range(3)]
p00 = 0 
p01 = 0
p02 = 0

for n in range(T-1):
    if Transitions[0][0]<= y[1][n] <=Transitions[0][1] and Transitions[0][0]<=y[1][n+1] <=Transitions[0][1]:
        p00+=1
    elif Transitions[0][0]<=y[1][n] <=Transitions[0][1] and Transitions[1][0]<=y[1][n+1] <=Transitions[1][1]:
        p01+=1
    elif Transitions[0][0]<=y[1][n] <=Transitions[0][1] and Transitions[2][0]<=y[1][n+1] <=Transitions[2][1]:
        p02+=1
p0 = p00+p01+p02       
[p00/p0, p01/p0, p02/p0]


# In[1344]:


plt.hist(y[1], bins = 25)


# In[873]:


int((0.5-5)/np.power(np.random.uniform(0,1), 1.0/1))


# In[872]:


(0.5-5)*np.power(np.random.uniform(0,1), 1.0/100000)


# In[1346]:


plt.hist(skewnorm.rvs(2, size = 1000))


# In[ ]:




