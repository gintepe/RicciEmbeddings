import numpy as np
from param_parser import parameter_parser
from embedding_clustering import create_and_run_model
import sys
import time

models = ['DeepWalk', 'DeepWalkWithRegularization', 'Ricci', 'GEMSECRicci', 'GEMSEC', 'GEMSECWithRegularization']

def get_report(res):
    report = ['average modularity {}'.format(np.mean(res)), 'standard deviation {}'.format(np.var(res))]
    return '\n'.join(report)


if __name__ == "__main__":
    args = parameter_parser()
    filename = args.input.split('/')[-1][:-4]
    with open("./res/{}{}.txt".format(filename, time.time()), "w") as file:
        for m in models:
            print('\n{}\n'.format(m))
            args.model = m
            modularities = []
            for i in range(10):
                print('trial {}'.format(i + 1))
                mod = create_and_run_model(args) 
                modularities.append(mod)
            file.write('\nModel: {} \n'.format(m))
            file.write('{}\n'.format(get_report(np.asarray(modularities))))
            file.write(str(modularities))

    