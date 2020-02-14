import numpy as np
from param_parser import parameter_parser
from embedding_clustering import create_and_run_model
from classification import classify, embed, select
import sys
import matplotlib.pyplot as plt
import time

# models = ['DeepWalk', 'DeepWalkWithRegularization', 'Ricci', 'GEMSECRicci', 'GEMSEC', 'GEMSECWithRegularization']
models = ['Ricci', 'GEMSECRicci']
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
tests = 10
iterations = 100

def get_report(res):
    report = ['average modularity {}'.format(np.mean(res)), 'standard deviation {}'.format(np.std(res))]
    return '\n'.join(report)

def get_single_info(row, lr):
    report = ['\n\nlearning rate: {}'.format(lr), 'average accuracy {}'.format(np.mean(row)), 'standard deviation {}\n'.format(np.std(row))]
    return '\n'.join(report)


# should I re-embed or just re-classify??
def loop_classify(args):
    with open("./res/{}{}.txt".format('cora', time.time()), "w") as file:
        for model in models:
            args.model = model
            print('model - {}'.format(model))
            file.write('\n Model: {}\n'.format(model))
            res = np.zeros((len(learning_rates), tests))
            for t in range(tests):
                embeddings, labels = embed(args)
                for i in range(len(learning_rates)):
                    lr = learning_rates[i]
                    train, train_labels, test, test_labels = select(embeddings, labels, 0.9)
                    res[i, t] = classify(train, train_labels, test, test_labels, args, iterations, lr)
            
            exps = np.arange(tests)
            for i in range(len(learning_rates)):
                row = res[i, :]
                lr = learning_rates[i]
                file.write(get_single_info(row, lr))
                file.write(str(row))
                plt.plot(exps, row, label='learning rate = {}'.format(lr))
            
            plt.title('Cora with {}'.format(model))
            plt.legend()
            plt.savefig('res/img/cora{}lr{}.png'.format(lr, model))
            plt.show()

def loop_embed(args):
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

if __name__ == "__main__":
    args = parameter_parser()
    # loop_classify(args)
    loop_embed(args)

    