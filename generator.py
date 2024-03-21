#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 00:18:51 2024

@author: vincenzopolitelli
"""

import networkx as nx
import numpy as np
from math import inf
import random
import matplotlib as mpl
# Use the pgf backend (must be done before import pyplot interface)
mpl.use('pgf')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "font.size": 9,
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })


############################ REMARK ############################
# In order to sample uniformly at random 
# labeled trees                 on n vertices we use the function nx.random_labeled_tree(n)
# rooted plane trees            --------------------------------- random_rooted_plane_tree(n)
# rooted unembedded binary tree --------------------------------- random_rooted_unembedded_binary_tree(n)
# rooted unembedded tree        --------------------------------- nx.random_unlabeled_rooted_tree(n)
# free tree                     --------------------------------- nx.random_unlabeled_tree(n)

### Rooted unembedded binary tree

cache={0:1, 1:1}

def c(n):
    """
    Generates the sequence (c_n)_{n >= 1} where c_n is the 
    number of binary unordered trees on n vertices
    """
    def c_aux(n):
        if n in cache:
            return cache[n]
        else:
            m=n//2
            res=0
            for k in range(m):
                res += c_aux(k)*c_aux(n-1-k)
            if n%2==1:
                res += (c_aux(m)*(c_aux(m) + 1))//2
            cache[n] = res
            return cache[n]
    return c_aux(n), cache

def random_rooted_unembedded_binary_tree(n=10):
    """
    
    Parameters
    ----------
    n : Integer
        Number of vertices of the tree. The default is 10.

    Returns
    A unlabeled binary tree choseu uniformly at random among
    the set of unlabeled binary trees with n vertices
    
    Remark: the root is constantly updated to be 0

    """
    
    if n == 1:
        t = nx.Graph()
        t.add_node(0)
        t.graph["root"] = 0
        return t
    
    c_n, cache = c(n)
    C_n = 0; proba = []
    k=n//2    
    for i in range(k): 
        C_n += cache[i]*cache[n - i - 1]
        proba.append(cache[i]*cache[n - i - 1])
    if n%2==1:
        C_n += cache[k]*(cache[k]+1)//2
        proba.append(cache[k]*(cache[k]+1)//2)

    proba = np.array(proba)
    proba = proba/C_n
    proba = np.array(proba,dtype='float64')
    
    num_right_nodes = np.random.choice(k + int(n%2==1), p=proba)
    if num_right_nodes == 0:
        t = random_rooted_unembedded_binary_tree(n-1)
        mapping = {n: n+1 for n in iter(t)}
        t = nx.relabel_nodes(t, mapping)
        t.add_node(0)
        t.add_edge(0,1)
        t.graph["root"]=0
        return t
    elif num_right_nodes == k:
        q = np.random.choice(2,p=[1/(cache[k]+1), 1 - 1/(cache[k]+1)])
        if q == 0:
            subtree = random_rooted_unembedded_binary_tree(num_right_nodes)
            return nx.join_trees([(subtree,subtree.graph["root"]), (subtree, subtree.graph["root"])])
    
    subtree1 = random_rooted_unembedded_binary_tree(num_right_nodes)
    subtree2 = random_rooted_unembedded_binary_tree(n-num_right_nodes-1)
    return nx.join_trees([(subtree1,subtree1.graph["root"]), (subtree2,subtree2.graph["root"])])


def DyckWordToTree(DyckWord, ones):
    """

    Parameters
    ----------
    DyckWord : array-like object
        A Dyck Word of +1s and -1s

    Returns
    -------
    The rooted plane tree associated to DyckWord

    """
    
    t = nx.DiGraph()
    t.add_node(0)
    t.graph["root"] = 0
    node_idx = 0
    curr_node = 0
    for i in DyckWord:
        if i == 1:
            node_idx += 1
            t.add_edge(curr_node, node_idx)
            curr_node = node_idx
        if i == -1:
            for edge in t.in_edges(curr_node):
                new_curr_node = edge[0]
            try:
                curr_node = new_curr_node
            except:
                print(DyckWord)
                print(ones)
                raise(ValueError)
    t = t.to_undirected()
    
    return t
            
def random_rooted_plane_tree(n):
    """

    Parameters
    ----------
    n : number of nodes

    Returns
    -------
    Returns a uniformly random Dyck word of length 2n.

    """
    
    ones = [+1 for _ in range(n)] + [-1 for _ in range(n-1)]
    random.shuffle(ones)
    # get minimum
    best_idx = 0
    lowest_point = 0
    cum_sum = 0
    for i in range(len(ones)):
        cum_sum += ones[i]
        if cum_sum <= lowest_point:
            best_idx = i
            lowest_point = cum_sum
    if best_idx==0:
        dyckWord = ones[1:]
    dyckWord = ones[best_idx +1 :] + ones[:best_idx + 1]
    return DyckWordToTree(dyckWord, ones)

### Drawing

def draw_tree(tree_generator, n=10, rooted=True):
    t = tree_generator(n)
    color_map = []
    for node in t:
        if rooted and node == t.graph["root"]:
            color_map.append("red")
        else:
            color_map.append("blue")
    nx.draw(t,node_color=color_map)
    return t

### Simulation

def large_voronoi_vector(G, agents):
    """

    Parameters
    ----------
    G : Undirected graph
    agents : distinct vertices of G

    Returns
    -------
    vor_vect : the corresponding Voronoi vectors

    """
    best_agents = {target: set() for target in G}
    for target in G:
        best_dist = inf
        for agent in agents:
            try:
                (dist, path) = nx.single_source_dijkstra(G, agent, target=target, cutoff=best_dist)
                if dist < best_dist:
                    best_agents[target] = {path[0]}
                    best_dist = dist
                elif dist == best_dist:
                    best_agents[target].add(path[0])
            except nx.NetworkXNoPath:
                continue
    vor_dict = {agent: 0 for agent in agents}
    for targ, aagents in best_agents.items():
        for only_agent in aagents:
            vor_dict[only_agent] += 1/len(aagents)
    vor_vect = np.array([val for val in vor_dict.values()])/G.number_of_nodes()
    return vor_vect

def voronoi_test(tree_generator,n,num_agents=2,m=1000):
    """

    Parameters
    ----------
    tree_generator : random tree generator 
                     from a given family of trees
    n : number of nodes of the generated tree
    num_agents : number of distinct agents to place 
                 on the vertices of the generated tree.
                 The default is 2.
    m : Number of iterations of the algorithm.
        The default is 1000.

    Returns
    -------
    p : p-value of the Kolmogorof-Smirnof test 
        of the num_agents * m elements of all the generated
        Voronoi vectors against a beta distribution of
        parameters 1, num_agents-1.
    
    Code used to generate Table 1 in the report.
    """
    
    if (tree_generator != nx.random_unlabeled_tree) and (tree_generator != nx.random_unlabeled_rooted_tree):
        tot_vor_vect = []
        for it in range(m):
            print(it)
            t = tree_generator(n)
            agents = set(np.random.choice(list(t.nodes),size=num_agents,replace=False))
            tot_vor_vect.append(large_voronoi_vector(t, agents)[1])
    
    else:
        tot_vor_vect = []
        tree_list = tree_generator(n, number_of_trees=m)
        for it in range(m):
            print(it)
            t = tree_list[it]
            agents = set(np.random.choice(list(t.nodes),size=num_agents,replace=False))
            tot_vor_vect.append(large_voronoi_vector(t, agents))
    
    tot_vor_vect= np.array(tot_vor_vect).ravel()
    np.save(f'arrays/tot_vor_vect__{tree_generator=}__{num_agents=}__{n=}', tot_vor_vect)
    if num_agents == 2:
        t_1, p= stats.kstest(tot_vor_vect, stats.uniform.cdf)
    else:
        alpha = np.ones(num_agents)
        t_2, p = stats.kstest(tot_vor_vect, stats.dirichlet.rvs(alpha, size=m * num_agents, random_state=2).ravel())
    return p


def p_values_array(tree_generator, filename, num_agents=2, m=1000):
    """

    Parameters
    ----------
    tree_generator : random tree generator 
                     from a given family of trees
    filename : File in which to save the returned array
    num_agents : number of distinct agents to place 
                 on the vertices of the generated tree.
                 The default is 2.
    m : Number of iterations of the algorithm.
        The default is 1000.

    Returns
    -------
    None.
    
    Code used to generate Figure 3 in the report.

    """
    p_vect = []
    for n in range(100, 1001, 50):
        print(f'--------------------- n = {n} --------------------')
        p_vect.append(voronoi_test(tree_generator, n, num_agents=num_agents, m=m))
    p_vect = np.array(p_vect)
    np.save(f'arrays/{filename}', p_vect)


def plot():
    """
    Plot used to generate Figure 2 in the report.

    """
    x = np.linspace(0, 1,1000)
    # Create a figure and subplots with a 3 by 2 layout
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    y=[[0 for _ in range(5)] for _ in range(4)]
    for num_agents in range(2, 6):
        y[num_agents-2][0]=np.load(f'arrays/tot_vor_vect__random_labeled_tree__{num_agents=}__n=200.npy')
        y[num_agents-2][1]=np.load(f'arrays/tot_vor_vect__random_rooted_plane_tree__{num_agents=}__n=200.npy')
        y[num_agents-2][2]=np.load(f'arrays/tot_vor_vect__random_rooted_unembedded_binary_tree__{num_agents=}__n=200.npy')
        y[num_agents-2][3]=np.load(f'arrays/tot_vor_vect__random_unlabeled_rooted_tree__{num_agents=}__n=200.npy')
        y[num_agents-2][4]=np.load(f'arrays/tot_vor_vect__random_unlabeled_tree__{num_agents=}__n=200.npy')
    
    # Plotting each subplot
    for i in range(0,2):
        frq, edges = np.histogram(y[0][i], bins=50, density=True)
        label_1 = 'EPDF for labeled trees' if i==0 else 'EPDF for rooted plane trees'
        axs[0, 0].plot(edges[:-1], frq, '-', label=label_1)
    label_2 = 'PDF of $\mathcal{B}(1,k-1)$'
    axs[0, 0].plot(x, stats.beta.pdf(x, 1, 1), 'r--',label=label_2)
    axs[0, 0].set_title('EPDF for $N=200$ and $k=2$')
    
    for i in range(0,2):
        frq, edges = np.histogram(y[1][i], bins=50, density=True)
        axs[0, 1].plot(edges[:-1], frq)
    axs[0, 1].plot(x, stats.beta.pdf(x, 1, 2), 'r--')
    axs[0, 1].set_title('EPDF for $N=200$ and $k=3$')
    
    for i in range(0,2):
        frq, edges = np.histogram(y[2][i], bins=50, density=True)
        axs[1, 0].plot(edges[:-1], frq)
    axs[1, 0].plot(x, stats.beta.pdf(x, 1, 3), 'r--')
    axs[1, 0].set_title('EPDF for $N=200$ and $k=4$')
    
    for i in range(0,2):
        frq, edges = np.histogram(y[3][i], bins=50, density=True)
        axs[1, 1].plot(edges[:-1], frq)
    axs[1, 1].plot(x, stats.beta.pdf(x, 1, 4), 'r--')
    axs[1, 1].set_title('EPDF for $N=200$ and $k=5$')
    
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    
    plt.savefig('epdf_plot.pdf', format='pdf')



