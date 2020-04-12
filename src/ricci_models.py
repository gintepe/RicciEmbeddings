import math
import time
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
from ricci_utils import ricci_curvature_weight_generator, ricci_curvature_weight_generator_precomputed, ricci_curvature_weight_generator_raw
from gemsec.layers import DeepWalker, Clustering, Regularization
from gemsec.calculation_helper import neural_modularity_calculator, classical_modularity_calculator
from gemsec.calculation_helper import gamma_incrementer, RandomWalker, SecondOrderRandomWalker
from gemsec.calculation_helper import index_generation, batch_input_generator, batch_label_generator
from gemsec.print_and_read import json_dumper, log_setup
from gemsec.print_and_read import initiate_dump_gemsec, initiate_dump_dw
from gemsec.print_and_read import tab_printer, epoch_printer, log_updater
from load_and_process import ricci_weights_reader
from gemsec.model import GEMSECWithRegularization


class DeepWalkWithRicci(GEMSECWithRegularization):
    """
    DeepWalk-Ricci algorithm implementation with Ricci.
    """
    def build(self):
        """
        Method to create the computational graph and initialize weights.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()+self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

        if self.args.ricci_weights == "Compute":
            self.weights = ricci_curvature_weight_generator(self.graph, self.args.ricci_transform_alpha)
        elif self.args.raw_ricci:
            self.weights = ricci_curvature_weight_generator_raw(self.graph, ricci_weights_reader(self.args.ricci_weights))
        else:
            self.weights = ricci_curvature_weight_generator_precomputed(self.graph, self.args.ricci_transform_alpha, ricci_weights_reader(self.args.ricci_weights))
            

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}

        return feed_dict

class GEMSECWithRicci(GEMSECWithRegularization):
    """
    GEMSEC class regularized by Ricci curvature.
    """
    def build(self):
        """
        Method to create the computational graph.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)
            self.cluster_layer = Clustering(self.args)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()
            self.loss = self.loss + self.gamma*self.cluster_layer(self.walker_layer)
            self.loss = self.loss + self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

        if self.args.ricci_weights == "Compute":
            self.weights = ricci_curvature_weight_generator(self.graph, self.args.ricci_transform_alpha)
        elif self.args.raw_ricci:
            self.weights = ricci_curvature_weight_generator_raw(self.graph, ricci_weights_reader(self.args.ricci_weights))
        else:
            self.weights = ricci_curvature_weight_generator_precomputed(self.graph, self.args.ricci_transform_alpha, ricci_weights_reader(self.args.ricci_weights))


    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}
        return feed_dict