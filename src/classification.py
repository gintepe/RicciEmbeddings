
# we want to embed the full network
# then seperate out the training and test sets
# and then evaluate

import pandas as pd
import networkx as nx
import numpy as np
from param_parser import parameter_parser
from model import GEMSECWithRegularization, GEMSEC, GEMSECWithRicci
from model import DeepWalkWithRegularization, DeepWalk, DeepWalkWithRicci

def cora_loader():
    path = 'cora/cora.cites'
    edges = pd.read_csv(path, sep='\t', header=None, names=['target', 'source'])
    names = ['id']
    for i in range(1433):
        names.append('f{}'.format(i))
    names.append('category')
    features = pd.read_csv(path, sep='\t', header=None, names=names])
    # you need to figure out a way to get the labels!!
    graph = nx.from_pandas_edgelist(edges)
    return nx.convert_node_labels_to_integers(graph)

def embed_and_select(args):
    graph = cora_loader()
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
    pivot = rount(embeddings.shape[0] * 0.9)
    np.random.shuffle(embeddings)
    train = embeddings
    print(embeddings.shape)
    

def process_label(label):
    labels = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    init_array = np.zeros((1, len(labels)))
        for i in range(len(labels)):
            if label == labels[i]:
                init_array[1, i] = 1
    return init_array



def classify(graph):
    pass

if __name__ == "__main__":
    args = parameter_parser()
    embed_and_select(args)

