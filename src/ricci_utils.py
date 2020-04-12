from GraphRicciCurvature.OllivierRicci import OllivierRicci
import math
import numpy as np
import networkx as nx
from tqdm import tqdm


def compute_ricci_curvature(graph):
    graph_copy = graph.copy()
    graph_copy.remove_edges_from(nx.selfloop_edges(graph_copy))
    orc = OllivierRicci(graph_copy, alpha=0.5)
    G = orc.compute_ricci_curvature()
    return G

def transform_ricci_curvature(curvature, alpha):
    return np.exp(alpha * curvature) / (1 + np.exp(alpha * curvature))

def calculate_weigth(edge, G, alpha):
    if edge[0] == edge[1]:
        return 0
    else:
        return transform_ricci_curvature(G[edge[0]][edge[1]]["ricciCurvature"], alpha)

def ricci_curvature_weight_generator(graph, alpha):
    """
    Function to retrieve the Ricci curvature for all of the edges.
    """
    print(" ")
    print("Ricci curvature calculation started.")
    print(" ")
    graph_copy = graph.copy()
    graph_copy.remove_edges_from(nx.selfloop_edges(graph_copy))
    orc = OllivierRicci(graph_copy, alpha=0.5)
    G = orc.compute_ricci_curvature()
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
    print(f"Generating transformed curvature weights, transformation alpha = {alpha}")
    return weights

def replace_if_self(e, curvatures):
    if e[0] == e[1]:
        return 0 
    else:
        if e in curvatures:
            return curvatures[e]
        else:
            return curvatures[(e[1], e[0])]

def ricci_curvature_weight_generator_raw(graph, curvatures):
    """
    Function to retrieve the Ricci curvature for all of the edges.
    """
    print(" ")
    print("Ricci curvature calculation started.")
    print(" ")
    edges = nx.edges(graph)
    weights = {e: replace_if_self(e, curvatures) for e in tqdm(edges)}
    weights_prime = {(e[1], e[0]): value for e, value in weights.items()}
    weights.update(weights_prime)
    print(" ")
    return weights
    print("Generating RAW curvature weights")

