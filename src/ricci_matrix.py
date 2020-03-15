from ricci_param_parser import parameter_parser
from print_and_read import graph_reader, ricci_weights_reader
from ricci_stuff import ricci_curvature_matrix_generator
from sklearn.decomposition import NMF
import networkx as nx
from calculation_helper import classical_modularity_calculator
from scipy.sparse import coo_matrix, csr_matrix
from test_looper import generate_gnp_graph


def get_matrix(graph, args, adjacency = False):
    if args.ricci_weights == 'Compute':
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha, produce_adjacency=adjacency)
    else:
        ricci_weights = ricci_weights_reader(args.ricci_weights)
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha, ricci_weights, produce_adjacency=adjacency)
    return ricci_matrix

def get_adj_matrix(graph):
    return nx.linalg.graphmatrix.adjacency_matrix(graph)

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

def do_things(args, use_ricci=True):
    graph = graph_reader(args.input)
    # graph = generate_gnp_graph(7000, 0.1)
    print(f'is this directed? {nx.is_directed(graph)}')
    compare_matrices(graph, args)
    # if use_ricci:
    #     matrix = get_matrix(graph, args)
    # else:
    #     print('no Ricci')
    #     matrix = get_adj_matrix(graph)
    
    matrix = get_matrix(graph, args, not use_ricci)

    print(type(matrix))
    print('original dimensions {}'.format(matrix.shape))
    print('percentage nonempty {}'.format(round(matrix.count_nonzero() * 100 / (matrix.shape[0] * matrix.shape[1]), 3)))
    w = get_decomposition(matrix, args.dimensions)
    print('Embedding dimensions {}'.format(w.shape))
    modularity, assignments = classical_modularity_calculator(graph, w, args)
    print('Modularity:')
    print(round(modularity, 3))
    # many many things seem to be getting assigned to the same cluster
    # print(assignments)

def get_decomposition(matrix, dims):
    nmf = NMF(dims)
    transformed = nmf.fit_transform(matrix)
    # though honestly I don't know what the transformation means here (if A = WH, then W)
    print(f'recostruction error {nmf.reconstruction_err_}')
    return transformed

if __name__ == "__main__":
    args = parameter_parser()
    do_things(args, use_ricci=True)

