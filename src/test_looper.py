import numpy as np
from param_parser import parameter_parser
from embedding_clustering import create_and_run_model
import sys

def report(res):
    print('average modularity {}'.format(np.mean(res)))
    print('standard deviation {}'.format(np.var(res)))
    print(res)


if __name__ == "__main__":
    args = parameter_parser()
    modularities = []
    for i in range(50):
        print('trial {}'.format(i + 1))
        mod = create_and_run_model(args)
        modularities.append(mod)
    report(np.asarray(modularities))

    