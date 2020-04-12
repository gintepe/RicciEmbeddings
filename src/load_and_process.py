import json
import pandas as pd
import numpy as np
import networkx as nx
import pickle

from constants import CAT, LABELS_CORA, FB_NET_PATH, FB_TARGET_PATH

def graph_reader(input_path):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def ricci_weights_reader(p):
    with open(p, 'rb') as handle:
        weights = pickle.loads(handle.read())
    return weights

def process_label(new_label, labels):
    num_labels = len(labels)
    init_array = np.zeros((1, num_labels))
    for i in range(num_labels):
        if new_label == labels[i]:
            init_array[0, i] = 1.0
    return init_array

def cora_loader():
    path = 'cora/cora.cites'
    edges = pd.read_csv(path, sep='\t', header=None, names=['target', 'source'])
    names = ['id']
    for i in range(1433):
        names.append('f{}'.format(i))
    names.append(CAT)
    path_features = 'cora/cora.content'
    features = pd.read_csv(path_features, sep='\t', header=None, names=names)
    # you need to figure out a way to get the labels!!
    graph = nx.from_pandas_edgelist(edges)
    for index, row in features.iterrows():
        graph.nodes[row['id']][CAT] = row[CAT]
    g = nx.convert_node_labels_to_integers(graph)
    one_hots = np.zeros((len(g.nodes()), len(LABELS_CORA)))
    for i in range(len(g.nodes())):
        one_hot = process_label(g.nodes[i][CAT], LABELS_CORA)
        one_hots[i, :] = one_hot
    return g, one_hots

def load_fb_net(categories_wanted):
    cat_col = "page_type"
    id_col = "id"
    graph = graph_reader(FB_NET_PATH)
    info = pd.read_csv(FB_TARGET_PATH)
    selected_categories_wanted = info[info.page_type.isin(categories_wanted)]
    selected_unwanted_categories = info[~info.page_type.isin(categories_wanted)]

    for index, row in selected_categories_wanted.iterrows():
        graph.nodes[row[id_col]][CAT] = row[cat_col]
    
    for index, row in selected_unwanted_categories.iterrows():
        graph.remove_node(row[id_col])

    largest_cc = max(nx.connected_components(graph), key=len)
    g = graph.subgraph(largest_cc).copy() 

    g = nx.convert_node_labels_to_integers(g)

    print(f'number of nodes {len(g.nodes)}, number of edges {len(g.edges)}, graph connected - {nx.is_connected(g)}')

    one_hots = np.zeros((len(g.nodes()), len(categories_wanted)))
    
    for i in range(len(g.nodes())):
        one_hot = process_label(g.nodes[i][CAT], categories_wanted)
        one_hots[i, :] = one_hot

    # print(f'number of nodes {len(g.nodes)}, number of edges {len(g.edges)}, graph connected - {nx.is_connected(g)}')
    return g, one_hots


def load_by_paths(edgelist_path, category_path):
    
    if "cora" in edgelist_path or "cora" in category_path:
        return cora_loader()

    one_hots = np.loadtxt(category_path)
    graph = graph_reader(edgelist_path)
    return graph, one_hots
