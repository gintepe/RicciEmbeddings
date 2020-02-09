
# we want to embed the full network
# then seperate out the training and test sets
# and then evaluate

import pandas as pd
import networkx as nx
import numpy as np
from param_parser import parameter_parser
from model import GEMSECWithRegularization, GEMSEC, GEMSECWithRicci
from model import DeepWalkWithRegularization, DeepWalk, DeepWalkWithRicci

CAT = 'category'
NUM_LABELS = 7
LABELS = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']


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
    one_hots = np.zeros((len(g.nodes()), NUM_LABELS))
    for i in range(len(g.nodes())):
        one_hot = process_label(g.nodes[i][CAT])
        one_hots[i, :] = one_hot
    return g, one_hots

def embed_and_select(args):
    graph, labels = cora_loader()
    if args.model == "GEMSECWithRegularization":
        print("GEMSECWithRegularization")
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        print("GEMSEC")
        model = GEMSEC(args, graph)
    elif args.model == "DeepWalkWithRegularization":
        print("DeepWalkWithRegularization")
        model = DeepWalkWithRegularization(args, graph)
    elif args.model == "Ricci":
        print("Ricci")
        model = DeepWalkWithRicci(args, graph)
    elif args.model == "GEMSECRicci":
        print("GEMSECRicci")
        model = GEMSECWithRicci(args, graph)
    else:
        print('DeepWalk')
        model = DeepWalk(args, graph)
    model.train()
    embeddings = model.final_embeddings
    # need to add the labels here
    pivot = round(embeddings.shape[0] * 0.9)
    p = np.random.permutation(embeddings.shape[0])
    train_idxs = p[:pivot]
    test_idxs = p[pivot:]
    train = embeddings[train_idxs]
    train_labels = labels[train_idxs]
    test = embeddings[test_idxs]
    test_labels = labels[test_idxs]
    print(train.shape)
    print(test.shape)
    print(embeddings.shape)
    return train, train_labels, test, test_labels
    

def process_label(label):
    init_array = np.zeros((1, NUM_LABELS))
    for i in range(NUM_LABELS):
        if label == LABELS[i]:
            init_array[0, i] = 1
    return init_array



def classify(graph):
    pass

if __name__ == "__main__":
    args = parameter_parser()
    graph = cora_loader()
    embed_and_select(args)

