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
from sklearn.cluster import KMeans




def show_results(G):
    # Print the first five results
    print("First 5 edges: ")
    for n1,n2 in list(G.edges())[:5]:
        print("Ollivier-Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2]["ricciCurvature"]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, "ricciCurvature").values()
    plt.hist(ricci_curvtures,bins=10)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)

    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=10)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights")

    plt.tight_layout()
    plt.show()

def my_spectral(graph):
    ad = nx.to_numpy_array(graph)
    a = manifold.spectral_embedding(ad, n_components=2)
    xs = a[:, 0]
    ys = a[:, 1]

    plt.scatter(xs, ys)

    for i in range(len(xs)):
        plt.annotate(i, (xs[i], ys[i]))

    plt.show()


def get_mds_coords(graph, n_dim = 2):
    distances = nx.floyd_warshall_numpy(graph)
    mds = manifold.MDS(dissimilarity='precomputed', n_components = n_dim)
    a = mds.fit_transform(distances)
    return a

def get_mds_layout(graph):
    coos = get_mds_coords(graph)
    return to_layout(graph, coos)

def get_spectral_coords_naiive(ad, n_dim = 2):
  return manifold.spectral_embedding(ad, n_components=n_dim)

# since this does laplacian eigenmaps and they favor large weights
def get_spectral_coords(ad, n_dim = 2, epsilon = 1e-14):
  max_val = np.amax(ad)
  # ad = (max_val + epsilon) - ad
  # ad = 1/(ad + epsilon)
  nonzero = np.nonzero(ad)
  ad[nonzero] = (max_val + epsilon) - ad[nonzero]

  # print(ad)
  return manifold.spectral_embedding(ad, n_components=n_dim)

def get_spectral_layout(ad, graph):
    coos = get_spectral_coords(ad)
    return to_layout(graph, coos)

def my_mds_paths(graph, coloring_attribute=None):
    ad = nx.to_numpy_array(graph)

    plt.figure(1, figsize=(15, 5))

    plt.subplot(131)
    plt.title('mds')

    lay_mds = get_mds_layout(graph)

    draw_with_layout(graph, layout=lay_mds, coloring_attribute=coloring_attribute)

    plt.subplot(132)
    plt.title('regular')

    draw_colored(graph, coloring_attribute=coloring_attribute)

    plt.subplot(133)
    plt.title('spectral')
    lay_sp = get_spectral_layout(ad, graph)

    draw_with_layout(graph, layout=lay_sp, coloring_attribute=coloring_attribute)

    plt.show()

def flow(graph, iterations):
    orf = OllivierRicci(graph, alpha=0.5, base=1, exp_power=0, proc=4, verbose="ERROR", weight="weight")
    G = orf.compute_ricci_flow(iterations=iterations)
    return G

def kmeans_cluster(G, clusters, dimensions = 2, state = None, embedding = 'mds', label = 'cluster'):
    if embedding == 'mds':
        coo = get_mds_coords(G, n_dim = dimensions)
    else:
        ad = nx.to_numpy_array(G)
        coo = get_spectral_coords(ad, n_dim = dimensions)
    kmeans = KMeans(n_clusters=clusters, random_state=state).fit(coo)
    l = kmeans.labels_
    for i in range(len(G.nodes)):
        G.nodes[i][label] = l[i]
    return G

# def kmeans_and_draw_mds(G, clusters, state):
#     G = kmeans_cluster(G, clusters, state)
#     mds_lay = to_layout(rnd_partition, coo)
#     draw_with_layout(rnd_partition, mds_lay, 'cluster')

def get_clusters(G, ground_truth = 'block', clustered = 'cluster'):
    gt, cl = [], []
    for i in range (len(G.nodes)):
        gt.append(G.nodes[i][ground_truth])
        cl.append(G.nodes[i][clustered])
    return gt, cl

def get_ami_score(G, ground_truth = 'block', cluster = 'cluster'):
    gt, cl = get_clusters(G, ground_truth, cluster)
    return adjusted_mutual_info_score(gt, cl, average_method='arithmetic')

def transform_to_numbered_communities(G):
    for i in range(len(G.nodes())):
        G.nodes()[i]['com'] = -1
    
    counter = 0
    for i in range(len(G.nodes())):
        to_increase = False
        community = G.nodes()[i]['community']
        for num in community:
            if G.nodes()[num]['com'] < 0:
                G.nodes()[num]['com'] = counter
                to_increase = True
        if to_increase:
            counter += 1
        
    return G

def flow_and_show(graph, iterations, color_attribute = None):
    orf = OllivierRicci(graph, alpha=0.5, base=1, exp_power=0, proc=4, verbose="ERROR", weight="weight")
    print("Applying ricci flow for {} iterations".format(iterations))
    plt.figure(1)
    G = orf.compute_ricci_flow(iterations=iterations)
    show_results(G)

    if color_attribute is not None:
        print('various projections of the original graph')
        my_mds_paths(graph, coloring_attribute=color_attribute)
        print('various projections of the graph after Ricci flow has been applied')
        my_mds_paths(G, coloring_attribute=color_attribute)
    else:
        print('various projections of the original graph')
        my_mds_paths(graph)
        print('various projections of the graph after Ricci flow has been applied')
        my_mds_paths(G)

    return G

def flow_and_cluster(G, iterations, cluster_gt, num_clusters):
    pre_cluster_mds = kmeans_cluster(G, num_clusters)
    pre_cluster_sp = kmeans_cluster(G, num_clusters, embedding='spectral', label = 'sp')
    pre_ami_mds = get_ami_score(pre_cluster_mds, ground_truth=cluster_gt)
    pre_ami_sp = get_ami_score(pre_cluster_sp, ground_truth=cluster_gt, cluster = 'sp')
    print("pre flow ami score for MDS {}\n pre flow ami score for spectral {} \n".format(pre_ami_mds, pre_ami_sp))
    # print("pre flow ami score for MDS {}".format(pre_ami_mds))

    post_flow = flow_and_show(G, iterations, color_attribute=cluster_gt)
    post_cluster_mds = kmeans_cluster(post_flow, num_clusters)
    post_cluster_sp = kmeans_cluster(post_flow, num_clusters, embedding = 'spectral', label = 'sp')
    post_ami_mds = get_ami_score(post_cluster_mds, ground_truth=cluster_gt)
    post_ami_sp = get_ami_score(post_cluster_sp, ground_truth=cluster_gt, cluster = 'sp')
    print("post flow ami score for MDS {}, difference (positive - improved, negative - worsened) {}\n post flow ami score for spectral {}, difference {}"
                .format(post_ami_mds, post_ami_mds - pre_ami_mds, post_ami_sp, post_ami_sp - pre_ami_sp))
    # print("post flow ami score for MDS {}, difference (positive - improved, negative - worsened) {}".format(post_ami_mds, post_ami_mds - pre_ami_mds))
    print('Colored according to K-means clusters')
    my_mds_paths(pre_cluster_mds, 'cluster')
    my_mds_paths(post_cluster_mds, 'cluster')

def flow_cluster_no_show(G, iterations, cluster_gt, num_clusters, dimensions = 2, print_info = False):
    pre_cluster_mds = kmeans_cluster(G, num_clusters, dimensions=dimensions)
    pre_cluster_sp = kmeans_cluster(G, num_clusters, embedding='spectral', label = 'sp', dimensions=dimensions)
    pre_ami_mds = get_ami_score(pre_cluster_mds, ground_truth=cluster_gt)
    pre_ami_sp = get_ami_score(pre_cluster_sp, ground_truth=cluster_gt, cluster = 'sp')

    post_flow = flow(G, iterations=iterations)
    post_cluster_mds = kmeans_cluster(post_flow, num_clusters, dimensions=dimensions)
    post_cluster_sp = kmeans_cluster(post_flow, num_clusters, embedding = 'spectral', label = 'sp', dimensions=dimensions)
    post_ami_mds = get_ami_score(post_cluster_mds, ground_truth=cluster_gt)
    post_ami_sp = get_ami_score(post_cluster_sp, ground_truth=cluster_gt, cluster = 'sp')
    if print_info:
        print("pre flow ami score for MDS {}\n pre flow ami score for spectral {} \n".format(pre_ami_mds, pre_ami_sp))
        print("pre flow ami score for MDS {}".format(pre_ami_mds))
        print("post flow ami score for MDS {}, difference (positive - improved, negative - worsened) {}\n post flow ami score for spectral {}, difference {}"
                .format(post_ami_mds, post_ami_mds - pre_ami_mds, post_ami_sp, post_ami_sp - pre_ami_sp))
        # print("post flow ami score for MDS {}, difference (positive - improved, negative - worsened) {}".format(post_ami_mds, post_ami_mds - pre_ami_mds))
    return pre_ami_mds, post_ami_mds, pre_ami_sp, post_ami_sp

def flow_cluster_no_show_vary_dimensions(G, iterations, cluster_gt, num_clusters, dimensions = [2], print_info = False):
    mds = {}
    sp = {}
    for dim in dimensions:
        mds_name = 'mds{}'.format(dim)
        sp_name = 'sp{}'.format(dim)
        pre_cluster_mds = kmeans_cluster(G, num_clusters, dimensions=dim, label = mds_name)
        pre_cluster_sp = kmeans_cluster(G, num_clusters, embedding='spectral', label = sp_name, dimensions=dim)
        pre_ami_mds = get_ami_score(pre_cluster_mds, ground_truth=cluster_gt, cluster = mds_name)
        pre_ami_sp = get_ami_score(pre_cluster_sp, ground_truth=cluster_gt, cluster = sp_name)
        mds[dim] = [pre_ami_mds]
        sp[dim] = [pre_ami_sp]

    post_flow = flow(G, iterations=iterations)

    for dim in dimensions:
        mds_name = 'mds{}'.format(dim)
        sp_name = 'sp{}'.format(dim)
        post_cluster_mds = kmeans_cluster(post_flow, num_clusters, dimensions = dim, label=mds_name)
        post_cluster_sp = kmeans_cluster(post_flow, num_clusters, embedding = 'spectral', dimensions = dim, label = sp_name)
        post_ami_mds = get_ami_score(post_cluster_mds, ground_truth=cluster_gt, cluster = mds_name)
        post_ami_sp = get_ami_score(post_cluster_sp, ground_truth=cluster_gt, cluster = sp_name)
        mds[dim].append(post_ami_mds)
        sp[dim].append(post_ami_sp)
        if print_info:
            print("MDS: pre flow {}, post flow: {}, difference (positive - improved) {}\nspectral: pre flow {}, post flow {}, difference {}"
                    .format(mds[dim][0], mds[dim][1], mds[dim][1] - mds[dim][0], sp[dim][0], sp[dim][1], sp[dim][1] - sp[dim][0]))
    return mds, sp

