from ricci_param_parser import parameter_parser
from load_and_process import graph_reader, ricci_weights_reader
from sklearn.decomposition import NMF
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
import community
from ricci_utils import calculate_weigth_from_precomputed, calculate_weigth, compute_ricci_curvature
from gemsec.calculation_helper import classical_modularity_calculator
from scipy.sparse import coo_matrix, csr_matrix
from constants import RICCI_MATRIX, HALF_RICCI_MATRIX, ADJACENCY_MATRIX, TRANSFORMATION_ALPHA, MATRIX_TYPES
    
def adjacency_matrix_generator(graph):
    xs = []
    ys = []
    data = []
    edges = nx.edges(graph)

    for e in edges:
        if not e[0] == e[1]:
            xs.append(e[0])
            ys.append(e[1])
            data.append(1.0)
        xs.append(e[1])
        ys.append(e[0])
        data.append(1.0)
    return coo_matrix((np.array(data), (np.array(xs), np.array(ys))))

def ricci_curvature_matrix_generator(graph, curvatures = None, full = True, transformation_alpha = TRANSFORMATION_ALPHA):
    
    xs = []
    ys = []
    data = []
    edges = nx.edges(graph)
    
    if curvatures is None:
        G = compute_ricci_curvature(graph)
 
    for e in edges:
        if curvatures is not None:
            ricci_weight = calculate_weigth_from_precomputed(e, curvatures, transformation_alpha)
        else:
            ricci_weight = calculate_weigth(e, G, transformation_alpha)
        xs.append(e[0])
        ys.append(e[1])
        data.append(ricci_weight)
        xs.append(e[1])
        ys.append(e[0])
        if full:
            data.append(ricci_weight)
        else:
            data.append(1.0)
        
    ricci_matrix = coo_matrix((np.array(data), (np.array(xs), np.array(ys))))
    return ricci_matrix

def get_matrix(graph, args):
    
    if args.matrix_type == ADJACENCY_MATRIX:
        return adjacency_matrix_generator(graph)
    elif args.matrix_type == RICCI_MATRIX:
        get_full_ricci_matrix = True
    else:
        get_full_ricci_matrix = False

    if args.ricci_weights == 'Compute':
        return ricci_curvature_matrix_generator(graph, full=get_full_ricci_matrix, transformation_alpha=args.ricci_transform_alpha)
    else:
        ricci_weights = ricci_weights_reader(args.ricci_weights)
        return ricci_curvature_matrix_generator(graph, ricci_weights, full=get_full_ricci_matrix, transformation_alpha=args.ricci_transform_alpha)

def get_adj_matrix(graph):
    return nx.linalg.graphmatrix.adjacency_matrix(graph)

def nmf_assignment_calculator(embeddings):
    max_indices = np.argmax(embeddings, axis=0).flatten()
    assignments = {i: int(max_indices[i]) for i in range(max_indices.shape[0])}
    return assignments

def compare_matrices(graph, args):
    # nx.draw(graph, with_labels = True)
    m1 = get_adj_matrix(graph)
    # print(m1)
    m2 = get_matrix(graph, args)
    # print(m2)
    m = m1 - m2
    print(f'number adj non-zero {m1.count_nonzero()}')
    print(f'number ricci adj non-zero {m2.count_nonzero()}')
    print(f'max element of diff {m.max()}')
    print(f'min element of diff {m.min()}')
    print(f'adj diagonal sum {m1.diagonal().sum()}')
    print(f'ricci adj diagonal sum {m2.diagonal().sum()}')
    print(f'number difference non-zero {m.count_nonzero()}')
    print(f'diagonal difference sum {m.diagonal().sum()}')
    print(f'total difference sum {m.sum()}')

def embed_and_get_modularity(args, clustering = 'kmeans'):
    graph = graph_reader(args.input)
    print(f'is this directed? {nx.is_directed(graph)}')
    # compare_matrices(graph, args)
    
    matrix = get_matrix(graph, args)

    print(type(matrix))
    print('original dimensions {}'.format(matrix.shape))
    print('percentage nonempty {}'.format(round(matrix.count_nonzero() * 100 / (matrix.shape[0] * matrix.shape[1]), 3)))
    W, H = get_decomposition(matrix, args.dimensions)
    print('Embedding dimensions {}'.format(W.shape))
    if clustering == "kmeans":
        modularity, assignments = classical_modularity_calculator(graph, W, args)
    else:
        assignments = nmf_assignment_calculator(H)
        modularity = community.modularity(assignments, graph)
    print('Modularity:')
    print(round(modularity, 3))

    return modularity

def get_decomposition(matrix, dims):
    nmf = NMF(dims)
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    print(f'recostruction error {nmf.reconstruction_err_}')
    return W, H

if __name__ == "__main__":
    args = parameter_parser()
    embed_and_get_modularity(args, clustering="max")

