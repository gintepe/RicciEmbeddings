import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import scipy as sp
from scipy.sparse import coo_matrix
from tqdm import tqdm

RICCI_ALPHA = 0.5
TRANSFORMATION_ALPHA = 4


def transform_ricci_curvature(curvature, alpha):
    return np.exp(alpha * curvature) / (1 + np.exp(alpha * curvature))

def calculate_weigth(edge, G, alpha):
    if edge[0] == edge[1]:
        return 0
    else:
        return transform_ricci_curvature(G[edge[0]][edge[1]]["ricciCurvature"], alpha)

def compute_ricci_curvature(graph):
    graph_copy = graph.copy()
    graph_copy.remove_edges_from(nx.selfloop_edges(graph_copy))
    orc = OllivierRicci(graph_copy, alpha=0.5)
    G = orc.compute_ricci_curvature()
    return G

def ricci_curvature_matrix_generator(graph, alpha, curvatures = None, produce_adjacency = False):
    
    xs = []
    ys = []
    data = []
    edges = nx.edges(graph)

    if produce_adjacency:
        for e in edges:
            if not e[0] == e[1]:
                xs.append(e[0])
                ys.append(e[1])
                data.append(1.0)
            xs.append(e[1])
            ys.append(e[0])
            data.append(1.0)
        return coo_matrix((np.array(data), (np.array(xs), np.array(ys))))
    
    if curvatures is None:
        G = compute_ricci_curvature(graph)
 
    for e in edges:
        if curvatures is not None:
            ricci_weight = calculate_weigth_from_precomputed(e, curvatures, TRANSFORMATION_ALPHA)
        else:
            ricci_weight = calculate_weigth(e, G, TRANSFORMATION_ALPHA)
        xs.append(e[0])
        ys.append(e[1])
        data.append(ricci_weight)
        xs.append(e[1])
        ys.append(e[0])
        # data.append(1.0)
        data.append(ricci_weight)
    ricci_matrix = coo_matrix((np.array(data), (np.array(xs), np.array(ys))))
    return ricci_matrix
    

def ricci_curvature_weight_generator(graph, alpha):
    """
    Function to retrieve the Ricci curvature for all of the edges.
    """
    print(" ")
    print("Ricci curvature calculation started.")
    print(" ")
    G = compute_ricci_curvature(graph)
    print("Curvature calculated")
    edges = nx.edges(graph)
    weights = {e: calculate_weigth(e, G, alpha) for e in tqdm(edges)}
    weights_prime = {(e[1], e[0]): value for e, value in weights.items()}
    weights.update(weights_prime)
    print(" ")
    return weights

def calculate_weigth_from_precomputed(edge, curvatures, alpha):
    if edge[0] == edge[1]:
        return 0
    else:
        return transform_ricci_curvature(curvatures[edge], alpha)

def ricci_curvature_weight_generator_precomputed(graph, alpha, curvatures):
    """
    Function to retrieve the Ricci curvature for all of the edges.
    """
    print(" ")
    print("Ricci curvature calculation started.")
    print(" ")
    edges = nx.edges(graph)
    weights = {e: calculate_weigth_from_precomputed(e, curvatures, alpha) for e in tqdm(edges)}
    weights_prime = {(e[1], e[0]): value for e, value in weights.items()}
    weights.update(weights_prime)
    print(" ")
    return weights