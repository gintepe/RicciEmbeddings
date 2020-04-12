import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from embedding_clustering_wrapper import create_and_run_model_extended as embed
from ricci_param_parser import parameter_parser
from graph_generators import generate_gnp_graph
from ricci_flow_explorations import flow
from constants import ORIG_ATTRIBUTE, RICCI_FLOW_METRIC, HOP_COUNT, SPECTRAL_EMBEDDING, ARGS_EMBEDDING



def get_distance(embedding, u, v):
    """
    Returns the euclidean distance between specified nodes u, v using the given embedding
    :param embedding: coordinates of embedded nodes
    :param u: index of node u
    :param v: index of node v
    """
    dist = np.linalg.norm(embedding[u, :] - embedding[v, :])
    return dist

def get_distance_vector(embedding, u, landmarks):
    """
    Returns a vector of distances from u to all landmarks
    :param embedding: ndarray, coordinates of embedded nodes
    :param u: index of node u
    :param landmarks: array of indices of landmark nodes
    """
    dists = np.zeros(landmarks.shape[0])
    for i in range(landmarks.shape[0]):
        dists[i] = get_distance(embedding, u, landmarks[i])

    # OR
    # diffs = embedding[landmarks, :] - embedding[u, :]
    # dists = np.linalg.norm(diffs, 2, axis = 1).flatten()

    return dists

def get_full_landmark_dist_embedding(embedding, landmarks):
    recovery_map = {}
    dists = np.zeros((embedding.shape[0] - landmarks.shape[0], landmarks.shape[0]))
    appended = 0
    for u in range(embedding.shape[0]):
        if not u in landmarks:
            dists[appended, :] = get_distance_vector(embedding, u, landmarks)
            recovery_map[appended] = u
            appended += 1

    return dists, recovery_map

def get_similarity_matrix(landmark_dists_1, landmark_dists_2):
    return euclidean_distances(landmark_dists_1, landmark_dists_2)

def determine_matching(similarity_matrix, use_hungarian = True):
    return linear_sum_assignment(similarity_matrix)

def matching_correct(u, v, ground_truth_correspondence):
    ### absolute
    return ground_truth_correspondence[u] == v

def get_accuracy(recovery_map_row, recovery_map_col, row_idxs, col_idxs, original_correspondence_map):
    """
    map is row -> col
    """
    #### absolute
    assert col_idxs.shape[0] <= row_idxs.shape[0], "columns should correspond to the subgraph"
    total = col_idxs.shape[0]
    matches = 0
    for i in range(total):
        if matching_correct(recovery_map_col[col_idxs[i]], recovery_map_row[row_idxs[i]], original_correspondence_map):
            matches += 1

    return matches/total


def remove_random_nodes(graph, removal_fraction):
    """
    Assumes the original graph is labelled by integers starting at 0
    """
    number_of_nodes = len(graph.nodes)
    new_graph = graph.copy()
    for i in range(number_of_nodes):
        new_graph.nodes[i][ORIG_ATTRIBUTE] = i

    number_to_remove = round(number_of_nodes * removal_fraction)
    nodes_to_remove =  np.random.permutation(number_of_nodes)[:number_to_remove]
    new_graph.remove_nodes_from(nodes_to_remove)
    new_graph = nx.convert_node_labels_to_integers(new_graph)

    original_mapping_full_to_removed = {}
    original_mapping_removed_to_full = {}
    for i in range(number_of_nodes - number_to_remove):
        original_mapping_removed_to_full[i] = new_graph.nodes[i][ORIG_ATTRIBUTE]
        original_mapping_full_to_removed[new_graph.nodes[i][ORIG_ATTRIBUTE]] = i

    return new_graph, original_mapping_full_to_removed, original_mapping_removed_to_full

def choose_landmarks(graph1, graph2, ground_truth_g2_to_g1, num_landmarks):
    landmarks_g2 = np.random.permutation(len(graph2.nodes))[:num_landmarks]
    landmarks_g1 = np.array([ground_truth_g2_to_g1[i] for i in landmarks_g2])

    return landmarks_g1, landmarks_g2

def get_embeddings(graph1, graph2, args):
    _, embedding_1 = np.array(embed(args, graph1))
    _, embedding_2 = np.array(embed(args, graph2))
    return embedding_1, embedding_2

def get_embeddings_spectral(graph1, graph2):
    embedding_1 = np.zeros((len(graph1.nodes), args.dimensions))
    embedding_2 = np.zeros((len(graph2.nodes), args.dimensions))

    sp_layout_1 = nx.spectral_layout(graph1, dim=args.dimensions)
    for i in range(len(graph1.nodes)):
        embedding_1[i, :] = np.array(sp_layout_1[i])
    
    sp_layout_2 = nx.spectral_layout(graph2, dim=args.dimensions)
    for i in range(len(graph2.nodes)):
        embedding_2[i, :] = np.array(sp_layout_2[i])
    return embedding_1, embedding_2
    

def align(graph1, graph2, ground_truth_g2_to_g1, num_landmarks, args, method = ARGS_EMBEDDING):
    assert len(graph1.nodes) >= len(graph2.nodes), "the graph to be aligned as a subgraph should be smaller"

    landmarks_1, landmarks_2 = choose_landmarks(graph1, graph2, ground_truth_g2_to_g1, num_landmarks)

    if method == RICCI_FLOW_METRIC:
        landmark_distances_1, idxs_to_nodes_1 = get_distance_matrix_ricci_flow(graph1, landmarks_1)
        landmark_distances_2, idxs_to_nodes_2 = get_distance_matrix_ricci_flow(graph2, landmarks_2)
    
    elif method == HOP_COUNT:
        landmark_distances_1, idxs_to_nodes_1 = get_hop_count_distances(graph1, landmarks_1)
        landmark_distances_2, idxs_to_nodes_2 = get_hop_count_distances(graph2, landmarks_2)
    
    else:
        if method == SPECTRAL_EMBEDDING:
            embedding_1, embedding_2 = get_embeddings_spectral(graph1, graph2)
        else:
            embedding_1, embedding_2 = get_embeddings(graph1, graph2, args)
        landmark_distances_1, idxs_to_nodes_1 = get_full_landmark_dist_embedding(embedding_1, landmarks_1)
        landmark_distances_2, idxs_to_nodes_2 = get_full_landmark_dist_embedding(embedding_2, landmarks_2)

    similarity_matrix = get_similarity_matrix(landmark_distances_1, landmark_distances_2)

    matching_g1_dist_idxs, matching_g2_dist_idxs = determine_matching(similarity_matrix)

    accuracy = get_accuracy(idxs_to_nodes_1, idxs_to_nodes_2, matching_g1_dist_idxs, matching_g2_dist_idxs, ground_truth_g2_to_g1)

    return accuracy

def shortest_path_distances_to_landmarks(graph, landmarks):
    distances = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(graph)[:, landmarks]
    distances_to_landmarks = np.zeros((len(graph.nodes) - len(landmarks), len(landmarks)))
    idx_to_node = {}
    non_landmark_counter = 0
    for i in range(distances.shape[0]):
        if not i in landmarks:
            distances_to_landmarks[non_landmark_counter, :] = distances[i, :]
            idx_to_node[non_landmark_counter] = i
            non_landmark_counter += 1
            
    return distances_to_landmarks, idx_to_node


def get_distance_matrix_ricci_flow(graph, landmarks):
    graph = flow(graph, iterations=100)
    return shortest_path_distances_to_landmarks(graph, landmarks)

def get_hop_count_distances(graph, landmarks):
    weight = "weight"
    nx.set_edge_attributes(graph, 1, weight)
    return shortest_path_distances_to_landmarks(graph, landmarks)

def test_alignment(args, graph, fraction_to_remove, num_landmarks, method = ARGS_EMBEDDING):
    distorted_graph, mapping_graph_to_distorted, mapping_distorted_to_graph = remove_random_nodes(graph, fraction_to_remove)
    acc = align(graph, distorted_graph, mapping_distorted_to_graph, num_landmarks, args, method=method)
    return acc

def test_with_gnp_graphs(tests, n, p, distortion_fraction, num_landmarks, args, method = ARGS_EMBEDDING):
    accuracy_results = []
    for i in range(tests):
        print(f'test {i}')
        graph = generate_gnp_graph(n, p)
        acc = test_alignment(args, graph, distortion_fraction, num_landmarks, method=method)
        print(f'accuracy obtained, with method {method}, {distortion_fraction} distortion, on a gnp graph with n = {n}, p = {p}, is {acc}')
        accuracy_results.append(acc)
    print("all accuracies")
    print(str(accuracy_results))

if __name__ == "__main__":
    
    N = 100
    P = 0.1
    DISTORTION = 0.1
    LANDMARKS = 5
    TESTS = 5

    args = parameter_parser()
    args.dimensions = 2
    args.cluster_number = 5
    test_with_gnp_graphs(TESTS, N, P, DISTORTION, LANDMARKS, args, method=ARGS_EMBEDDING)


