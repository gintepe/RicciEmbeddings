import networkx as nx
import numpy as np
from embedding_clustering_wrapper import create_and_run_model_extended as embed
from ricci_param_parser import parameter_parser
from visualisations import to_layout, draw_with_layout, draw_color_edges
from ricci_utils import compute_ricci_curvature
import matplotlib.pyplot as plt
from ricci_flow_explorations import flow

if __name__ == "__main__":
    
    karate = nx.karate_club_graph()
    args = parameter_parser()

    # args.dimensions = 2
    # args.cluster_number = 2

    # args.model = "Ricci"
    # _, embedding = embed(args, karate)
    # embedding = np.array(embedding)
    # layout_dw = to_layout(karate, embedding)
    
    # draw_with_layout(karate, layout_dw, 'club')
    # plt.show()

    # args.model = "GEMSECRicci"
    # _, embedding = embed(args, karate)
    # embedding = np.array(embedding)
    # layout_dwr = to_layout(karate, embedding)

    # draw_with_layout(karate, layout_dwr, 'club')

    # print(args.raw_ricci)

    # plt.show()
    G = compute_ricci_curvature(karate)
    pos = nx.spring_layout(G)
    draw_color_edges(pos, "ricciCurvature", G, node_color_attribute='club')

    G = flow(G, 10)
    draw_color_edges(pos, "ricciCurvature", G, width_attribute="weight", node_color_attribute='club')
