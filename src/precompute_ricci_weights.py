from calculation_helper import ricci_curvature_weight_generator
import pickle
import sys
from print_and_read import graph_reader
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

def get_curvatures_as_dict(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    G = orc.compute_ricci_curvature()
    curvatures = {e: G[e[0]][e[1]]["ricciCurvature"] for e in G.edges()}
    return curvatures


if __name__ == "__main__":
    f = "data/" + sys.argv[1] + "_edges.csv"
    target_name = "data/ricci/" + sys.argv[1] + "_curvatures.txt"
    G = graph_reader(f)
    # weights = ricci_curvature_weight_generator(G, 4)
    # with open(target_name, 'wb') as handle:
    #     pickle.dump(weights, handle)
    curvatures = get_curvatures_as_dict(G)
    with open(target_name, 'wb') as handle:
        pickle.dump(curvatures, handle)
