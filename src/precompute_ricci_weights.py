from calculation_helper import ricci_curvature_weight_generator
import pickle
import sys
from print_and_read import graph_reader
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from classification import cora_loader
import time

def get_curvatures_as_dict(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    # orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO", method="ATD")
    s = time.time()
    G = orc.compute_ricci_curvature()
    print ('time for curvature computation: {}'.format(time.time() - s))
    curvatures = {e: G[e[0]][e[1]]["ricciCurvature"] for e in G.edges()}
    return curvatures

def precompute_cora():
    G, oh = cora_loader()
    print(len(G.nodes()))
    print(len(G.edges()))
    target_name = 'cora/cora_curvatures.txt'
    curvatures = get_curvatures_as_dict(G)
    with open(target_name, 'wb') as handle:
        pickle.dump(curvatures, handle)

if __name__ == "__main__":
    f = "data/" + sys.argv[1] + "_edges.csv"
    target_name = "data/ricci/" + sys.argv[1] + "_ATDcurvatures.txt"
    G = graph_reader(f)
    curvatures = get_curvatures_as_dict(G)
    with open(target_name, 'wb') as handle:
        pickle.dump(curvatures, handle)
    # precompute_cora()
