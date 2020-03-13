import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Facebook politicians network.
    """
    parser = argparse.ArgumentParser(description="Run GEMSEC.")

    parser.add_argument("--input",
                        nargs="?",
                        default="./data/politician_edges.csv",
	                help="Input graph path.")

    # parser.add_argument("--embedding-output",
    #                     nargs="?",
    #                     default="./output/embeddings/politician_embedding.csv",
	#                 help="Embeddings path.")

    # parser.add_argument("--cluster-mean-output",
    #                     nargs="?",
    #                     default="./output/cluster_means/politician_means.csv",
	#                 help="Cluster means path.")

    # parser.add_argument("--log-output",
    #                     nargs="?",
    #                     default="./output/logs/politician.json",
	#                 help="Log path.")

    # parser.add_argument("--assignment-output",
    #                     nargs="?",
    #                     default="./output/assignments/politician.json",
	#                 help="Log path.")

    # parser.add_argument("--dump-matrices",
    #                     type=bool,
    #                     default=True,
	#                 help="Save the embeddings to disk or not. Default is not.")

    # parser.add_argument("--model",
    #                     nargs="?",
    #                     default="GEMSECWithRegularization",
	#                 help="The model type.")
    
    parser.add_argument("--cluster-number",
                        type=int,
                        default=20,
	                help="Number of clusters. Default is 20.")

    parser.add_argument("--t-alpha",
                        type=float,
                        default=4,
	                help="Ricci curvature weight transformation hyperparameter. Default is 4.")

    parser.add_argument("--ricci-alpha",
                        type=float,
                        default=0.5,
	                help="Ricci curvature computation hyperparameter. Default is 0.5.")


    parser.add_argument("--dimensions",
                        type=int,
                        default=16,
	                help="Number of dimensions to collapse to. Default is 16.")

    parser.add_argument("--ricci-weights",
                        nargs="?",
                        default="Compute",
	                help="Path to read the ricci curvature weights from")

    
    
    return parser.parse_args()