import networkx as nx
import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import logging
from sklearn import manifold
from visualisations import to_layout, draw_colored, draw_with_layout
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from ricci_flow_explorations import flow_cluster_no_show, flow_cluster_no_show_vary_dimensions, transform_to_numbered_communities
from graph_generators import safe_generate_lfr

def report_mean_std(lst, name, f = None):
    a = np.array(lst)
    mean = np.mean(a)
    std_dev = np.std(a)
    str = 'For {}, the mean is {} and std deviation is {}, std error is {}'.format(name, mean, std_dev, std_dev / (np.sqrt(len(lst))))
    if f is None:
        print(str)
    else:
        f.write(f'\n{str}\n')
    return mean, std_dev

def plot_diffs(diffs, n_test, emb_name, save_name):
    exps = np.arange(n_test)
    plt.plot(exps, diffs, 'bo', color='xkcd:lightish blue', label='differences')
    plt.legend()
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Experiment')
    plt.ylabel('AMI score delta')
    plt.title(f'Difference under {emb_name} embedding')
    plt.savefig(save_name)
    plt.clf()
    plt.cla()
    plt.close()

def test_gaussian_partition(num_nodes, avg_cluster, shape, p_in, p_out, flow_iterations, number_of_tests, exp_id=None):
    org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami, diff_mds, diff_sp = [], [], [], [], [], []
    for i in range (number_of_tests):
        print('test {}'.format(i))
        G = nx.generators.community.gaussian_random_partition_graph(num_nodes, avg_cluster, shape, p_in, p_out, directed=False).to_undirected()
        expected_num_clusters = num_nodes // avg_cluster
        oma, fma, osa, fsa = flow_cluster_no_show(G, flow_iterations, 'block', expected_num_clusters)
        print('flow done')
        org_mds_ami.append(oma)
        flow_mds_ami.append(fma)
        org_sp_ami.append(osa)
        flow_sp_ami.append(fsa)
        diff_mds.append(fma - oma)
        diff_sp.append(fsa - osa)
        print('diff on MDS {}'.format(fma-oma))
        print('diff on Spectral {}'.format(fsa - osa))

    with open(f"./res/reruns/gnp_exp{exp_id}_0409.txt", "w") as f:
        print(f'Experiment {exp_id} completed, saving information')
        f.write('\nExperiments run {} times'.format(number_of_tests))
        report_mean_std(org_mds_ami, 'original graphs under MDS', f)
        report_mean_std(flow_mds_ami, 'graphs after Ricci flow under MDS', f)
        report_mean_std(diff_mds, 'differences using MDS', f)
        report_mean_std(org_sp_ami, 'original graphs under spectral embedding', f)
        report_mean_std(flow_sp_ami,  'graphs after Ricci flow under spectral embedding', f)
        report_mean_std(diff_sp, 'differences using spectral embedding', f)

    plot_diffs(np.array(flow_mds_ami) - np.array(org_mds_ami), number_of_tests, "MDS", f"./res/reruns/gnp_exp{exp_id}_MDS_0409.png")
    plot_diffs(np.array(flow_sp_ami) - np.array(org_sp_ami), number_of_tests, "spectral", f"./res/reruns/gnp_exp{exp_id}_sp_0409.png")

    return org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami

def test_gaussian_partition_vary_dimension(num_nodes, avg_cluster, shape, p_in, p_out, flow_iterations, number_of_tests, dimensions):
    org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami, diff_mds, diff_sp = {d: [] for d in dimensions} , {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}
    for i in range (number_of_tests):
        print('test {}'.format(i))
        G = nx.generators.community.gaussian_random_partition_graph(num_nodes, avg_cluster, shape, p_in, p_out, directed=False).to_undirected()
        expected_num_clusters = num_nodes // avg_cluster
        mds, sp = flow_cluster_no_show_vary_dimensions(G, flow_iterations, 'block', expected_num_clusters, dimensions=dimensions)
        print('flow done')
        for dim in dimensions:
            org_mds_ami[dim].append(mds[dim][0])
            flow_mds_ami[dim].append(mds[dim][1])
            org_sp_ami[dim].append(sp[dim][0])
            flow_sp_ami[dim].append(sp[dim][1])
            diff_mds[dim].append(mds[dim][1] - mds[dim][0])
            diff_sp[dim].append(sp[dim][1] - sp[dim][0])
    print('\nExperiments run {} times'.format(number_of_tests))
    for dim in dimensions:
        print('embedding dimensions = {}'.format(dim))
        report_mean_std(diff_mds[dim], 'differences using MDS')
        report_mean_std(diff_sp[dim], 'differences using spectral embedding')

    return org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami

def test_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, flow_iterations, number_of_tests, exp_id=None):
    org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami, diff_mds, diff_sp = [], [], [], [], [], []
    for i in range (number_of_tests):
        print(f'Test {i}\n')
        G_init = safe_generate_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, 0, 30)
        print('LFR graph generated {}'.format(i))
        G = transform_to_numbered_communities(G_init)
        expected_num_clusters = num_nodes // min_communities
        oma, fma, osa, fsa = flow_cluster_no_show(G, flow_iterations, 'com', expected_num_clusters)
        org_mds_ami.append(oma)
        flow_mds_ami.append(fma)
        org_sp_ami.append(osa)
        flow_sp_ami.append(fsa)
        diff_mds.append(fma - oma)
        print(f'diff under MDS {fma - oma}, {oma}, {fma}')
        diff_sp.append(fsa - osa)
        print(f'diff under Spectral {fsa - osa}, {osa}, {fsa}')
    print('\nExperiments run {} times'.format(number_of_tests))
    with open(f"./res/reruns/lfr_exp{exp_id}.txt", "w") as f:
        f.write('\nExperiments run {} times'.format(number_of_tests))
        report_mean_std(org_mds_ami, 'original graphs under MDS', f)
        report_mean_std(flow_mds_ami, 'graphs after Ricci flow under MDS', f)
        report_mean_std(diff_mds, 'differences using MDS', f)
        report_mean_std(org_sp_ami, 'original graphs under spectral embedding', f)
        report_mean_std(flow_sp_ami,  'graphs after Ricci flow under spectral embedding', f)
        report_mean_std(diff_sp, 'differences using spectral embedding', f)  

    plot_diffs(np.array(flow_mds_ami) - np.array(org_mds_ami), number_of_tests, "MDS", f"./res/reruns/lfr_exp{exp_id}_MDS.png")
    plot_diffs(np.array(flow_sp_ami) - np.array(org_sp_ami), number_of_tests, "spectral", f"./res/reruns/lfr_exp{exp_id}_sp.png")

    return org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami

def test_lfr_vary_dimensions(num_nodes, avg_degree, min_communities, mu, tau1, tau2, flow_iterations, number_of_tests, dimensions):
    org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami, diff_mds, diff_sp = {d: [] for d in dimensions} , {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}, {d: [] for d in dimensions}
    for i in range (number_of_tests):
        print('test {}'.format(i))
        G_init = safe_generate_lfr(num_nodes, avg_degree, min_communities, mu, tau1, tau2, 0, 30)
        print('LFR graph generated {}'.format(i))
        G = transform_to_numbered_communities(G_init)
        expected_num_clusters = num_nodes // min_communities
        mds, sp = flow_cluster_no_show_vary_dimensions(G, flow_iterations, 'com', expected_num_clusters, dimensions=dimensions)
        print('flow done')
        for dim in dimensions:
            org_mds_ami[dim].append(mds[dim][0])
            flow_mds_ami[dim].append(mds[dim][1])
            org_sp_ami[dim].append(sp[dim][0])
            flow_sp_ami[dim].append(sp[dim][1])
            diff_mds[dim].append(mds[dim][1] - mds[dim][0])
            diff_sp[dim].append(sp[dim][1] - sp[dim][0])
    print('\nExperiments run {} times'.format(number_of_tests))
    for dim in dimensions:
        print('embedding dimensions = {}'.format(dim))
        report_mean_std(diff_mds[dim], 'differences using MDS')
        report_mean_std(diff_sp[dim], 'differences using spectral embedding')

    return org_mds_ami, flow_mds_ami, org_sp_ami, flow_sp_ami

config_gp_1 = {
    "num_nodes": 50,
    "avg_cluster": 10,
    "shape":10,
    "p_in":0.6,
    "p_out":0.2,
    "flow_iterations": 30,
    "number_of_tests": 30,
    "exp_id": 1,
}

config_gp_2 = {
    "num_nodes": 50,
    "avg_cluster": 10,
    "shape": 10,
    "p_in": 0.6,
    "p_out": 0.3,
    "flow_iterations": 30,
    "number_of_tests": 30,
    "exp_id": 2,
}

config_gp_3 = {
    "num_nodes": 80,
    "avg_cluster": 10,
    "shape": 10,
    "p_in": 0.6,
    "p_out": 0.2,
    "flow_iterations": 30,
    "number_of_tests": 30,
    "exp_id": 3,
}

config_lfr_1 = {
    "num_nodes": 100,
    "avg_degree": 13,
    "min_communities": 10,
    "mu": 0.3, 
    "tau1": 2, 
    "tau2": 1.2, 
    "flow_iterations": 30, 
    "number_of_tests": 30,
    "exp_id": 1,
}

config_lfr_2 = {
    "num_nodes": 100,
    "avg_degree": 10,
    "min_communities": 15,
    "mu": 0.3, 
    "tau1": 3, 
    "tau2": 2, 
    "flow_iterations": 30, 
    "number_of_tests": 30,
    "exp_id": 2,
}

config_lfr_3 = {
    "num_nodes": 150,
    "avg_degree": 15,
    "min_communities": 20,
    "mu": 0.3, 
    "tau1": 3, 
    "tau2": 2, 
    "flow_iterations": 30, 
    "number_of_tests": 30,
    "exp_id": 3,
}

if __name__ == "__main__":
    print("TESTING GNP GRAPHS")

    for config in [config_gp_1, config_gp_2, config_gp_3]:
        test_gaussian_partition(**config)

    # print("TESTING LFR NETS")

    # for config in [config_lfr_1, config_lfr_2, config_lfr_3]:
    #     test_lfr(**config)
