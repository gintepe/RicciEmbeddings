"""Running the model."""

from gemsec.param_parser import parameter_parser
from gemsec.print_and_read import graph_reader
from gemsec.model import GEMSECWithRegularization, GEMSEC
from gemsec.model import DeepWalkWithRegularization, DeepWalk

def create_and_run_model(args, graph = None):
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
    else:
        print('DeepWalk')
        model = DeepWalk(args, graph)
    model.train()

    return model

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
