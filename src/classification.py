
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

def embed(args):
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
    return embeddings, labels

def select(embeddings, labels, train_fraction):
    pivot = round(embeddings.shape[0] * train_fraction)
    p = np.random.permutation(embeddings.shape[0])
    train_idxs = p[:pivot]
    test_idxs = p[pivot:]
    train = embeddings[train_idxs]
    train_labels = labels[train_idxs]
    test = embeddings[test_idxs]
    test_labels = labels[test_idxs]
    return train, train_labels, test, test_labels


def embed_and_select(args):
    embeddings, labels = embed(args)
    train, train_labels, test, test_labels = select(embeddings, labels, 0.9)
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
    cost_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model), reduction_indices=[1]))
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

def do_the_thing(args):
    xtrain, ytrain, xtest, ytest = embed_and_select(args)
    classify(xtrain, ytrain, xtest, ytest, args, 100, 0.01)

if __name__ == "__main__":
    args = parameter_parser()
    # graph = cora_loader()
    # embed_and_select(args)
    do_the_thing(args)

