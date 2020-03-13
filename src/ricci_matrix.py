from ricci_param_parser import parameter_parser
from print_and_read import graph_reader, ricci_weights_reader
from ricci_stuff import ricci_curvature_matrix_generator
from sklearn.decomposition import NMF
from calculation_helper import classical_modularity_calculator


def get_matrix(graph, args):
    if args.ricci_weights == 'Compute':
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha)
    else:
        ricci_weights = ricci_weights_reader(args.ricci_weights)
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha, ricci_weights)
    return ricci_matrix

def do_things(args):
    graph = graph_reader(args.input)
    ricci_matrix = get_matrix(graph, args)
    print('original dimensions {}'.format(ricci_matrix.shape))
    print('percentage nonempty {}'.format(round(ricci_matrix.count_nonzero() * 100 / (ricci_matrix.shape[0] * ricci_matrix.shape[1]), 3)))
    w = get_decomposition(ricci_matrix, args.dimensions)
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
    return transformed

if __name__ == "__main__":
    args = parameter_parser()
    do_things(args)

