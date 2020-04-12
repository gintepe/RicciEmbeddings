import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def to_layout(graph, coordinates):
    lay = {}
    for i in range(len(graph.nodes())):
        node = list(graph.nodes)[i]
        lay[node] = coordinates[i, :]

    return lay

def draw_with_layout(graph, layout, coloring_attribute = None):
    if coloring_attribute is not None:
        groups = nx.get_node_attributes(graph, coloring_attribute).values()
        viridis = cm.get_cmap('viridis', len(set(groups)))
        # color_list = plt.cm.get_cmap('viridis')[np.linspace(0, 1, len(groups))]
        color_list = viridis.colors

        color_dict = dict(zip(set(groups), color_list))
    if coloring_attribute is not None:
        nx.draw_networkx(graph, pos=layout, with_labels=True, nodelist=graph.nodes(),
                    node_color=[color_dict[x] for x in groups],
                    alpha=0.8)
    else:
        nx.draw_networkx(graph, pos=layout, with_labels=True)

def draw_colored(graph, coloring_attribute = None):
    if coloring_attribute is not None:
        groups = nx.get_node_attributes(graph, coloring_attribute).values()
        viridis = cm.get_cmap('viridis', len(set(groups)))
        color_list = viridis.colors
        color_dict = dict(zip(set(groups), color_list))
        # color_list = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        # color_dict = dict(zip(groups, color_list))
    if coloring_attribute is not None:
        nx.draw_networkx(graph, nodelist=graph.nodes(),
                    node_color=[color_dict[x] for x in groups],
                    alpha=0.8)
    else:
        nx.draw_networkx(graph)

def draw_color_edges(pos, edge_color_attribute, G, width_attribute = None, node_color_attribute = None):
    edges,weights = zip(*nx.get_edge_attributes(G,edge_color_attribute).items())
    fig, axs = plt.subplots()
    cmap = plt.cm.coolwarm
    vmin = -0.75
    vmax = 0.75

    width = []
    for edge in edges:
        if width_attribute is not None:
            width.append(G.get_edge_data(*edge)[width_attribute]*2)
        else:
            width.append(2.0)

    node_colors = ['b' for x in G.nodes()]
    if node_color_attribute is not None:    
        groups = nx.get_node_attributes(G, node_color_attribute).values()
        viridis = cm.get_cmap('cividis', len(set(groups)))
        color_list = viridis.colors
        color_dict = dict(zip(set(groups), color_list))
        node_colors = [color_dict[x] for x in groups]

    nx.draw(G, pos, node_color=node_colors, node_size = 100, edgelist=edges, edge_color=weights, width = width, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))

    axins1 = inset_axes(axs,
                    width="50%",  # width = 50% of parent_bbox width
                    height="2.5%",  # height : 5%
                    loc='lower center')

    fig.colorbar(sm, orientation="horizontal", cax=axins1, label = "Curvature Values")
    plt.show()