
# we want to embed the full network
# then seperate out the training and test sets
# and then evaluate

import pandas as pd
import networkx as nx
import numpy as np
from param_parser import parameter_parser
from model import GEMSECWithRegularization, GEMSEC, GEMSECWithRicci
from model import DeepWalkWithRegularization, DeepWalk, DeepWalkWithRicci
import tensorflow as tf

WEIGHT_ATTRIBUTE = "weight"
REWEIGHT_VALUE = 10
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

def embed(args, graph):
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
    return embeddings

def embed_and_load_with_reweight(args, train_fraction, test_fraction, reweight_value=REWEIGHT_VALUE):
    graph, labels = cora_loader()
    num_vertices = graph.number_of_nodes()
    train_idxs, test_idxs = select_indices(num_vertices, train_fraction, test_fraction)
    reweight_by_selection(graph, train_idxs, reweight_value)
    
    args.ricci_weights = "Compute"
    embeddings = embed(args, graph)

    return embeddings, labels, train_idxs, test_idxs


def embed_and_load(args, reweight=False):
    graph, labels = cora_loader()
    embeddings = embed(args, graph)
    return embeddings, labels

def select(embeddings, labels, train_fraction, test_fraction=None):
    train_idxs, test_idxs = select_indices(embeddings.shape[0], train_fraction, test_fraction=test_fraction)
    return get_sets_for_classification(embeddings, labels, train_idxs, test_idxs)

def get_sets_for_classification(embeddings, labels, train_idxs, test_idxs):
    train = embeddings[train_idxs]
    train_labels = labels[train_idxs]
    test = embeddings[test_idxs]
    test_labels = labels[test_idxs]
    return train, train_labels, test, test_labels

def select_indices(num_vertices, train_fraction, test_fraction = None):
    """ Assumes train_fraction + test_fraction <= 1"""
    pivot = round(num_vertices * train_fraction)
    p = np.random.permutation(num_vertices)
    train_idxs = p[:pivot]
    if not test_fraction is None:
        pivot2 = round(num_vertices * (1 - test_fraction))
        test_idxs = p[pivot2:]
    else:
        test_idxs = p[pivot:]

    print(f"\n\ntrain number{len(train_idxs)}, test number {len(test_idxs)}\n\n")

    return train_idxs, test_idxs


def reweight_by_selection(G, vertices_selected, reweight_value):
    """Assumes an unweighted graph is passed in"""
    nx.set_edge_attributes(G, 1, WEIGHT_ATTRIBUTE)
    for v in vertices_selected:
        for neighbour in G[v]:
            G[v][neighbour][WEIGHT_ATTRIBUTE] = reweight_value


def embed_and_select(args, train_fraction, test_fraction=None, reweight = False, reweight_value=REWEIGHT_VALUE):
    if reweight:
        embeddings, labels, train_idxs, test_idxs = embed_and_load_with_reweight(args, train_fraction=train_fraction, test_fraction=test_fraction, reweight_value=reweight_value)
        train, train_labels, test, test_labels = get_sets_for_classification(embeddings, labels, train_idxs, test_idxs)
    else :
        embeddings, labels = embed_and_load(args)
        train, train_labels, test, test_labels = select(embeddings, labels, train_fraction=train_fraction, test_fraction=test_fraction)
    return train, train_labels, test, test_labels
    

def process_label(label):
    init_array = np.zeros((1, NUM_LABELS))
    for i in range(NUM_LABELS):
        if label == LABELS[i]:
            init_array[0, i] = 1.0
    return init_array

# for each embedding sweep over learning rates

def classify(xtrain, ytrain, xtest, ytest, args, iterations, learning_rate):

    training_iteration = iterations
    learning_rate = learning_rate
    display_step = 99

    x = tf.placeholder(tf.float32,[None, args.dimensions])
    W = tf.Variable(tf.zeros([args.dimensions, NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.placeholder(tf.float32,[None, NUM_LABELS])

    # Minimize error using cross entropy
    # Cross entropy
    # cost_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model), reduction_indices=[1]))
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for iteration in range(training_iteration):
            avg_cost = 0.

            batch_xs = xtrain
            batch_ys = ytrain
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})

            # Display logs per eiteration step
            if iteration % display_step == 0:
                print ("Iteration:" +  '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

        print ("Tuning completed!")

        # Test the model
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        a = accuracy.eval({x: xtest, y: ytest})
        print ("Accuracy: {}".format(a))
    
    return a

def embed_and_predict(args, train_fraction, test_fraction=None, reweight=False, reweight_value=REWEIGHT_VALUE):
    xtrain, ytrain, xtest, ytest = embed_and_select(args, train_fraction, test_fraction, reweight=reweight, reweight_value=reweight_value)
    acc = classify(xtrain, ytrain, xtest, ytest, args, 100, 0.1)
    return acc

import matplotlib.pyplot as plt
import time
import datetime

if __name__ == "__main__":
    args = parameter_parser()
    embed_and_predict(args, train_fraction=0.9, test_fraction=None, reweight=False, reweight_value=4)
    # train_fracs = [0.05, 0.1, 0.2]
    # rew_values = [1, 2, 5.5, 8, 16]
    # # rew_values = [1/16, 1/8, 1/4, 1/2, 1]
    # te_f = 0.1
    # smoothing_trials = 3
    # t = datetime.datetime.now()
    # with open("./res/{}{}.txt".format('cora_rew_varied', t), "w") as file:
    #     file.write(f'Model - {args.model}\nTrials for each combination - {smoothing_trials}\n')
    #     answers = {}
    #     file.write(f'reweightings - {str(rew_values)}\n')
    #     for tr_f in train_fracs:
    #         vals = []
    #         for rv in rew_values:
    #             accs = []
    #             for i in range(smoothing_trials):
    #                 print(f'\n\nreweight to {rv}, trial {i}\n\n')
    #                 acc = embed_and_predict(args, train_fraction=tr_f, test_fraction=te_f, reweight=True, reweight_value=rv)
    #                 accs.append(acc)
    #             vals.append(np.mean(np.array(accs)))
    #         file.write(f'\nFor training fraction {tr_f} and test fraction {te_f}:\n')
    #         file.write(str(vals))
    #         answers[tr_f] = vals
    # for tr_f in train_fracs:
    #     plt.plot(rew_values, answers[tr_f], label=f'training fraction = {tr_f}', linestyle='--', marker='x')
    # plt.title(f'Dependency of accuracy on reweighting value, model - {args.model}')#
    # plt.legend()
    # plt.savefig(f'res/img/cora_rewlow_dep_{args.model}{t}.png')
    # plt.show()

