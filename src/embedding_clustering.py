"""Running the model."""

from param_parser import parameter_parser
from print_and_read import graph_reader
from model import GEMSECWithRegularization, GEMSEC, GEMSECWithRicci
from model import DeepWalkWithRegularization, DeepWalk, DeepWalkWithRicci

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    graph = graph_reader(args.input)
    if args.model == "GEMSECWithRegularization":
        print("GEMSECWithRegularization")
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        print("GEMSEC")
        model = GEMSEC(args, graph)
    elif args.model == "DeepWalkWithRegularization":
        print("DeepWalkWithRegularization")
        model = DeepWalkWithRegularization(args, graph)
    elif args.model == "Ricci":
        print("Ricci")
        model = DeepWalkWithRicci(args, graph)
    elif args.model == "GEMSECRicci":
        print("GEMSECRicci")
        model = GEMSECWithRicci(args, graph)
    else:
        print('DeepWalk')
        model = DeepWalk(args, graph)
    model.train()
    return model.modularity_score

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
