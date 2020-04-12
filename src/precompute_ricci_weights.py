from ricci_utils import ricci_curvature_weight_generator
import pickle
import sys
from load_and_process import graph_reader
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import time
import matplotlib.pyplot as plt
import collections

def get_curvatures_as_dict(graph, method = "OTD"):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO", method=method)
    s = time.time()
    G = orc.compute_ricci_curvature()
    time_elapsed = time.time() - s
    print ('time for curvature computation: {}'.format(time_elapsed))
    curvatures = {e: G[e[0]][e[1]]["ricciCurvature"] for e in G.edges()}
    return curvatures, time_elapsed

def degree_hist(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

def precompute_cora():
    G, oh = cora_loader()
    print(len(G.nodes()))
    print(len(G.edges()))
    target_name = 'cora/cora_curvatures.txt'
    curvatures, _ = get_curvatures_as_dict(G)
    with open(target_name, 'wb') as handle:
        pickle.dump(curvatures, handle)

def precompute_by_path(p, save_name, method = "OTD"):
    target_name = "data/ricci/" + save_name + f"{method}_curvatures.txt"
    G = graph_reader(p)
    print(len(G.nodes), len(G.edges))
    print(max(G.nodes), min(G.nodes))
    # degree_hist(G)
    curvatures, _ = get_curvatures_as_dict(G, method=method)
    save_curvatures(target_name, curvatures)


def save_curvatures(filename, curvatures):
    with open(filename, 'wb') as handle:
        pickle.dump(curvatures, handle)


if __name__ == "__main__":
    f = "/home/ginte/dissertation/diss/data/fb_class/fb_company_tvshow.csv"
    # f = "/home/ginte/dissertation/diss/data/politician_edges.csv"
    precompute_by_path(f, "fb_company_tvshow", "OTD")
    # precompute_by_path(f, sys.argv[1])
    # precompute_cora()
