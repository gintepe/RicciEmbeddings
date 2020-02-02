from ricci_param_parser import parameter_parser
from print_and_read import graph_reader, ricci_weights_reader
from ricci_stuff import ricci_curvature_matrix_generator

def do_things(args):
    graph = graph_reader(args.input)
    if args.ricci_weights == 'Compute':
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha)
    else:
        ricci_weights = ricci_weights_reader(args.ricci_weights)
        ricci_matrix = ricci_curvature_matrix_generator(graph, args.t_alpha, ricci_weights)
    print(ricci_matrix.shape)
    print(ricci_matrix.count_nonzero())
    print(ricci_matrix.count_nonzero() * 100 / (ricci_matrix.shape[0] * ricci_matrix.shape[1]))

if __name__ == "__main__":
    args = parameter_parser()
    do_things(args)
