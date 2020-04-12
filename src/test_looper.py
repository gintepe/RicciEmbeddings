import numpy as np
from ricci_param_parser import parameter_parser as ricci_param_parser
from embedding_clustering_wrapper import create_and_run_model_extended as create_and_run_model
from classification import classify, embed_and_load, embed_and_select, select
import sys
import matplotlib.pyplot as plt
import networkx as nx
import time
import datetime
from precompute_ricci_weights import get_curvatures_as_dict, save_curvatures
from ricci_matrix import embed_and_get_modularity
from constants import MATRIX_TYPES, RICCI_MATRIX
from graph_generators import generate_graphs
from load_and_process import graph_reader

models_full = ['DeepWalk', 'DeepWalkWithRegularization', 'Ricci', 'GEMSECRicci', 'GEMSEC', 'GEMSECWithRegularization']
# models_full = ['DeepWalk', 'DeepWalkWithRegularization']
models = ['Ricci', 'GEMSECRicci']
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
# learning_rates = [0.01, 0.1]
TESTS = 10
iterations = 100

graphs = ['data/tvshow_edges.csv', 'data/politician_edges.csv', 'data/government_edges.csv', 'data/company_edges.csv', 'data/public_figure_edges.csv', 'data/athletes_edges.csv']
atd_curvatures = ['data/ricci/tvshow_ATDcurvatures.txt', 'data/ricci/politician_ATDcurvatures.txt', 'data/ricci/government_ATDcurvatures.txt', 'data/ricci/company_ATDcurvatures.txt', 'data/ricci/public_figure_ATDcurvatures.txt', 'data/ricci/athletes_ATDcurvatures.txt']
curvatures =['data/ricci/tvshow_curvatures.txt', 'data/ricci/politician_curvatures.txt', 'data/ricci/government_curvatures.txt', 'data/ricci/company_curvatures.txt', 'data/ricci/public_figure_curvatures.txt', 'data/ricci/athletes_curvatures.txt']


def get_report(res):
    report = ['average modularity {}'.format(np.mean(res)), 'standard deviation {}'.format(np.std(res))]
    return '\n'.join(report)

def get_single_info(row, lr):
    report = ['\n\nlearning rate: {}'.format(lr), 'average accuracy {}'.format(np.mean(row)), 'standard deviation {}\n'.format(np.std(row))]
    return '\n'.join(report)

def get_time():
    return datetime.datetime.now()

def save_model(file, learning_rates, res):
    for i in range(len(learning_rates)):
        row = res[i, :]
        lr = learning_rates[i]
        file.write(get_single_info(row, lr))
        file.write(str(row))

def plot_results_vertical(learning_rates, res_by_model, tests, name):
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        for j in range(len(models_full)):
            m =  models_full[j]
            val = np.ones(tests) * (j+1)
            row = res_by_model[m][i, :]
            plt.plot(val, row, 'o', alpha=0.6, label=f'model = {m}')
        plt.title(f'{name} with learning rate {lr}')
        # plt.legend()
        xs = np.arange(len(models_full)) + 1
        plt.xticks(xs, models_full, rotation = 60)
        plt.subplots_adjust(bottom=0.4)
        plt.savefig('res/img/{}_lr{}_{}.png'.format(name, lr, get_time()))
        plt.show()

def get_name(args):
    if "cora" in args.input or "cora" in args.classes:
        name = "cora"
    else:
        name = "fb_net"

    return name

def loop_classify(args, train_frac, test_frac=None, learning_rates = LEARNING_RATES, tests = TESTS):
    
    name = f"{get_name(args)}_train_{train_frac}_test_{test_frac}"

    with open("./res/{}{}.txt".format(name, get_time()), "w") as file:
        file.write(f"training set fraction is {train_frac}, test set fraction is {test_frac}")
        ress = {}
        for model in models_full:
            args.model = model
            print('model - {}'.format(model))
            file.write('\n\nModel: {}\n'.format(model))
            res = np.zeros((len(learning_rates), tests))
            for t in range(tests):
                embeddings, labels = embed_and_load(args)
                for i in range(len(learning_rates)):
                    lr = learning_rates[i]
                    train, train_labels, test, test_labels = select(embeddings, labels, train_frac, test_frac)
                    res[i, t] = classify(train, train_labels, test, test_labels, args, iterations, lr)
            
            ress[model] = res

            exps = np.arange(tests)
            save_model(file, learning_rates, res)

    plot_results_vertical(learning_rates, ress, tests, name)  

def loop_classify_reweightings(args, train_frac, test_frac, reweight_value, learning_rates = LEARNING_RATES, seperate = False, tests = TESTS):

    args.raw_ricci = False
    
    name = get_name(args)

    with open("./res/{}_rew{}_sep{}_{}.txt".format(name, reweight_value, seperate, get_time()), "w") as file:
        
        file.write(f"training set fraction is {train_frac}, test set fraction is {test_frac}, reweight value is {reweight_value}")
        
        ress = {}
        for model in models_full:
            args.model = model
            print('model - {}'.format(model))
            file.write('\n\nModel: {}\n'.format(model))
            res = np.zeros((len(learning_rates), tests))
            for t in range(tests):
                for i in range(len(learning_rates)):
                    train, train_labels, test, test_labels = embed_and_select(args, train_frac, test_frac, reweight=True, seperate=True, reweight_value=reweight_value)
                    lr = learning_rates[i]
                    res[i, t] = classify(train, train_labels, test, test_labels, args, iterations, lr)
            
            
            ress[model] = res

            save_model(file, learning_rates, res)

    plot_results_vertical(learning_rates, ress, tests, f'{name}_rew{reweight_value}_sep{seperate}')

def test_reweighted_embedding(args, values, train_frac, test_frac, learning_rates=LEARNING_RATES, seperate = False, tests = TESTS):
    for rev_value in values:
        loop_classify_reweightings(args, train_frac, test_frac, rev_value, learning_rates=learning_rates, seperate=seperate, tests=tests)

def loop_embed(args, tests, models, name_prefix = ""):
    filename = args.input.split('/')[-1][:-4]
    with open("./res/{}_{}_{}.txt".format(filename, name_prefix, get_time()), "w") as file:
        for m in models:
            print('\n{}\n'.format(m))
            args.model = m
            modularities = []
            times = []
            for i in range(tests):
                print('trial {}'.format(i + 1))
                t = time.time()
                mod, _ = create_and_run_model(args) 
                times.append(time.time() - t)
                modularities.append(mod)
            file.write('\nModel: {} \n'.format(m))
            file.write('{}\n'.format(get_report(np.asarray(modularities))))
            file.write(str(modularities))
            file.write('\nTimes:\n')
            file.write(str(times))

def lambda_tunning_raw(args, tests, graph_paths, curvature_paths, lambdas, models):
    args.raw_ricci = True
    for l in lambdas:
        args.lambd = l
        for graph_path, curvature_path in zip(graph_paths, curvature_paths):
            args.input = graph_path
            args.ricci_weights = curvature_path
            loop_embed(args, tests, models, name_prefix=f"RAW_lambda_{l}")

def matrix_embed(args, f, clustering, tests, extra_info = ""):
    f.write(extra_info)
    modularities = []
    times = []
    for i in range(tests):
        print('trial {}'.format(i + 1))
        t = time.time()
        mod = embed_and_get_modularity(args, clustering=clustering) 
        times.append(time.time() - t)
        modularities.append(mod)
    f.write('{}\n'.format(get_report(np.asarray(modularities))))
    f.write(str(modularities))
    f.write('\nTimes:\n')
    f.write(str(times))

def loop_matrix_embed(args, tests, clustering = 'kmeans'):
    filename = args.input.split('/')[-1][:-4]
    with open("./res/matrix/{}_{}_{}_matrix.txt".format(clustering, filename, get_time()), "w") as f:
        for matrix_type in MATRIX_TYPES:
            args.matrix_type = matrix_type
            matrix_embed(args, f, clustering, tests, f'\nMatrix being used: {matrix_type}\n')

def matrix_test_tranformation_alpha(alphas, args, tests, clustering = 'kmeans'):
    for i in range(len(graphs)):
        print('\n\n\nRunning embedding with non negative matrix factorization on graph {}\n\n\n'.format(graphs[i]))
        args.input = graphs[i]
        args.ricci_weights = curvatures[i]
        args.matrix_type = RICCI_MATRIX
        filename = args.input.split('/')[-1][:-4]
        with open("./res/matrix/alpha_tuning/high{}_{}_{}_matrix.txt".format(clustering, filename, get_time()), "w") as f:
            for alpha in alphas:
                args.ricci_transform_alpha = alpha
                matrix_embed(args, f, clustering, tests, f'\n\nMatrix used - {RICCI_MATRIX}, transformation alpha value {alpha} \n')
            

def generate_graphs_curvatures_prefixes(node_numbers, graph_type, distinguisher = ''):
    name_prefs, c_times = generate_graphs(node_numbers, graph_type=graph_type, name_addon=distinguisher)
    with open(f"./res/runtimes/{graph_type}_{node_numbers[0]}_{node_numbers[-1]}_{distinguisher}_curvature_times.txt", 'w') as f:
        f.write(f'Graphs generated with nodes {str(node_numbers)}, of type {graph_type}\n')
        f.write('Times taken for curvature calculations:\n')
        f.write(str(c_times))
        f.write('\nFilename prefixes:')
        f.write(str(name_prefs))
    return name_prefs

def get_graph_paths(prefs):
    graph_paths = []
    curvature_paths = []
    for pref in prefs:
        graph_paths.append(f'{pref}.edgelist')
        curvature_paths.append(f'{pref}_curvatures.txt')
    return graph_paths, curvature_paths

def runtime_test_flexible(node_numbers = None, graph_paths = None, curvature_paths = None, model = 'Ricci', graph_type='gnp', distinguisher = ''):
    if node_numbers is not None:
        name_prefs = generate_graphs_curvatures_prefixes(node_numbers, graph_type, distinguisher)
        graph_paths, curvature_paths = get_graph_paths(name_prefs)
    runtime_test(graph_paths=graph_paths, curvature_paths=curvature_paths, model=model, graph_type=graph_type, distinguisher=distinguisher)

def runtime_test(node_numbers = None, graph_paths = None, curvature_paths = None, model = 'Ricci', graph_type='gnp', distinguisher = ''):

    if node_numbers is None and (graph_paths is None or curvature_paths is None):
        raise Exception("graphs and curvatures need to be either generated from sizes or given as paths")

    f_name = f"./res/runtimes/{graph_type}_{model}{get_time()}.txt"
    args.model = model
    embed_times = []

    # if node_numbers is not None:
    #     name_prefs = generate_graphs_curvatures_prefixes(node_numbers, graph_type)
            
    #     graph_paths, curvature_paths = get_graph_paths(name_prefs)
    
    node_numbers = []
    edge_numbers = []
    with open(f_name, "w") as file:
        file.write(f"Model - {model}, graphs - {graph_type}, node values:\n")

        for i in range(len(graph_paths)):
            args.ricci_weights = curvature_paths[i]
            graph = nx.read_edgelist(graph_paths[i], nodetype = int)
            
            n = len(graph.nodes())
            m = len(graph.edges())
            node_numbers.append(n)
            edge_numbers.append(m)
            
            if n < 20:
                args.cluster_number = n

            print(f'number of nodes - {n}, number of edges - {m}')
            print('running embedding')
            t_start = time.time()
            mod, _ = create_and_run_model(args, graph=graph)
            t_end = time.time()
            embed_times.append(t_end - t_start)

        file.write(str(node_numbers))
        file.write('\nEdge counts:\n')
        file.write(str(edge_numbers))

        file.write('\nObserved embedding (no Ricci curvature computation) runtimes:\n')
        file.write(str(embed_times))
        
def test_matrix_embedding(graphs, curvatures, tests, clustering = 'kmeans'):
    for i in range(len(graphs)):
        print('\n\n\nRunning embedding with non negative matrix factorization on graph {}\n\n\n'.format(graphs[i]))
        args.input = graphs[i]
        args.ricci_weights = curvatures[i]
        loop_matrix_embed(args, tests, clustering=clustering)

def test_runtimes_from_prefixes(prefs, models, graph_type):
    graph_paths = []
    curvature_paths = []
    for pref in prefs:
        graph_paths.append(f'{pref}.edgelist')
        curvature_paths.append(f'{pref}_curvatures.txt')

    for model in models_full:
        runtime_test(graph_paths=graph_paths, curvature_paths=curvature_paths, model=model, graph_type=graph_type)

def test_runtimes_from_sizes(node_numbers, graph_type, models):
    prefs = generate_graphs_curvatures_prefixes(node_numbers, graph_type, distinguisher='')
    test_runtimes_from_prefixes(prefs, models, graph_type)

def summarize_graph(path):
    graph = graph_reader(path)
    print(path + '\n')
    n = len(graph.nodes)
    m = len(graph.edges)
    density = m/ ( n * (n-1) / 2)
    print(f"The number of nodes is {n}, number of edges is {m} and density is {density}\n")
    print(nx.info(graph))

if __name__ == "__main__":
    # for g in graphs:
    #     summarize_graph(g)
    # args = parameter_parser()
    # lambdas = [2.0**-5, 2.0**-4, 2.0**-3, 2.0**-2, 2.0**-1]
    # lambda_tunning_raw(args, 10, graphs, curvatures, lambdas, models)

    args = ricci_param_parser()
    # loop_classify(args, 0.05, 0.2, learning_rates=[0.01, 0.1, 0.05], tests=10)

    # matrix_test_tranformation_alpha([64, 128, 256], args, 10, clustering = 'max')
    args.dimensions = 20
    test_matrix_embedding(graphs, curvatures, 10, clustering='max')
    # test_reweighted_embedding(args, [2, 8, 32, 64, 128], 0.1, 0.2, learning_rates=[0.01, 0.1], seperate=True)
    # args.input = "/home/ginte/dissertation/diss/data/fb_class/fb_company_tvshow.csv"
    # args.classes = "/home/ginte/dissertation/diss/data/fb_class/fb_company_tvshow_classes.txt"
    # test_reweighted_embedding(args, [64, 128], 0.02, 0.1, learning_rates=[0.1], seperate=True)

    

    # loop_classify(args, 0.9, learning_rates=[0.1, 0.01])
    # loop_classify_reweightings(args, 0.1, 0.2, 8)
    # loop_embed(args)
    # pows_of_2 = [16, 32, 64, 128, 512, 1024]

    # runtime_test(node_numbers=pows_of_2, model='Ricci', graph_type='lfr')

    # # prefs = ['./data/for_runtimes/160.1gnp', './data/for_runtimes/320.1gnp', './data/for_runtimes/640.1gnp', './data/for_runtimes/1280.1gnp', './data/for_runtimes/5120.1gnp', './data/for_runtimes/10240.1gnp']
    # prefs = ['./data/for_runtimes/160.1lfr', './data/for_runtimes/320.1lfr', './data/for_runtimes/640.1lfr', './data/for_runtimes/1280.1lfr', './data/for_runtimes/5120.1lfr', './data/for_runtimes/10240.1lfr']
    # # # atd_prefs = ['./data/for_runtimes/160.1gnp_ATD', './data/for_runtimes/320.1gnp_ATD', './data/for_runtimes/640.1gnp_ATD', './data/for_runtimes/1280.1gnp_ATD', './data/for_runtimes/5120.1gnp_ATD', './data/for_runtimes/10240.1gnp_ATD']
    # test_runtimes_from_prefixes(prefs, models_full, 'lfr')
    
