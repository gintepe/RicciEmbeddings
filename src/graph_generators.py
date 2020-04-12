import networkx as nx
from precompute_ricci_weights import get_curvatures_as_dict, save_curvatures

def generate_gnp_graph(n, p):
    connected = False
    while not connected:
        g = nx.fast_gnp_random_graph(n, p)
        connected = nx.is_connected(g)
        if not connected:
            print('disconnected graph, trying again')
    return g

def generate_lfr_graph(n):
    p = 0.1
    connected = False
    tau1 = 2
    tau2 = 1.5
    mu = 0.3
    avg_degree = 4#p * (n - 1) / 2
    max_degree = 10#n
    while not connected:
        print("generating LFR graph failed, trying again")
        g = None
        try:
            g = nx.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=avg_degree, min_degree=None, max_degree=max_degree, min_community=None, max_community=None, tol=1e-07, max_iters=1000, seed=None)
        except:
            print("LFR generation failed")
        if not g is None:
            connected = nx.is_connected(g)
    return g

def generate_graphs(node_numbers, graph_type='gnp', name_addon = ''):
    name_prefs = []
    curvature_times = []
    p = 0.1
    for n in node_numbers:
        # generate graph
        print('generating graph')
        if graph_type == 'lfr':
            graph = generate_lfr_graph(n)
        else:
            graph = generate_gnp_graph(n, p)

        fname_prefix = f"./data/for_runtimes/{n}{p}{graph_type}{name_addon}"
        nx.write_edgelist(graph, f"{fname_prefix}.edgelist", data=False)
        name_prefs.append(fname_prefix)

        print('Computing curvature')    
        curvatures, c_time = get_curvatures_as_dict(graph, method="OTD")
        curvature_times.append(c_time)
        save_curvatures(f"{fname_prefix}_curvatures.txt", curvatures)

    return name_prefs, curvature_times

def safe_generate_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, attempt, max_attempts):
    print('attempt {}'.format(attempt))
    if attempt >= max_attempts:
        raise Exception('Maximum attempts to generate a connected LFR graph with the given settings exceeded')
    G = None
    try:
        G = nx.LFR_benchmark_graph(n=num_nodes, tau1=tau1, tau2=tau2, mu=mu, average_degree=avg_degree, min_degree=None, max_degree=None, min_community=min_communities, max_community=None, tol=1e-07, max_iters=500, seed=None)
    except:
        print('generating the graph failed, trying again')
    if G is None:
        return safe_generate_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, attempt + 1, max_attempts)
    if nx.is_connected(G):
        return G
    else:
        print('generated graph is disconnected, trying again')
        return safe_generate_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, attempt + 1, max_attempts)