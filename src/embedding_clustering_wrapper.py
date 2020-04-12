from ricci_param_parser import parameter_parser
from gemsec.embedding_clustering import create_and_run_model
from ricci_models import DeepWalkWithRicci, GEMSECWithRicci
from load_and_process import graph_reader

def create_and_run_model_extended(args, graph = None):
    
    if graph is None:
        graph = graph_reader(args.input)
    
    if args.model == "Ricci":
        print("Ricci")
        model = DeepWalkWithRicci(args, graph)
        model.train()
    elif args.model == "GEMSECRicci":
        print("GEMSECRicci")
        model = GEMSECWithRicci(args, graph)
        model.train()
    else:
        model = create_and_run_model(args)
    
    return model.modularity_score, model.final_embeddings

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model_extended(args)