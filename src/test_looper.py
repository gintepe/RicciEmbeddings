import numpy as np
from param_parser import parameter_parser
from embedding_clustering import create_and_run_model
from classification import classify, embed_and_load, embed_and_select, select
import sys
import matplotlib.pyplot as plt
import time
import datetime

models_full = ['DeepWalk', 'DeepWalkWithRegularization', 'Ricci', 'GEMSECRicci', 'GEMSEC', 'GEMSECWithRegularization']
# models_full = ['DeepWalk', 'DeepWalkWithRegularization']
models = ['Ricci', 'GEMSECRicci']
# learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
learning_rates = [0.01, 0.1]
tests = 10
iterations = 100

graphs = ['data/tvshow_edges.csv', 'data/politician_edges.csv', 'data/government_edges.csv', 'data/company_edges.csv', 'data/public_figure_edges.csv', 'data/athletes_edges.csv']
atd_curvatures = ['data/ricci/tvshow_ATDcurvatures.txt', 'data/ricci/politician_ATDcurvatures.txt', 'data/ricci/government_ATDcurvatures.txt', 'data/ricci/company_ATDcurvatures.txt', 'data/ricci/public_figure_ATDcurvatures.txt', 'data/ricci/athletes_ATDcurvatures.txt']
curvatures =['data/ricci/tvshow_curvatures.txt', 'data/ricci/politician_curvatures.txt', 'data/ricci/government_curvatures.txt', 'data/ricci/company_curvatures.txt', 'data/ricci/public_figure_curvatures.txt', 'data/ricci/athletes_curvatures.txt']


def get_report(res):
    report = ['average modularity {}'.format(np.mean(res)), 'standard deviation {}'.format(np.std(res))]
    return '\n'.join(report)

def get_single_info(row, lr):
    report = ['\n\nlearning rate: {}'.format(lr), 'average accuracy {}'.format(np.mean(row)), 'standard deviation {}\n'.format(np.std(row))]
    return '\n'.join(report)

def get_time():
    return datetime.datetime.now()

# just re-classify??
def loop_classify(args,train_frac, test_frac=None):
    with open("./res/{}{}.txt".format('cora', get_time()), "w") as file:
        file.write(f"training set fraction is {train_frac}, test set fraction is {test_frac}")
        for model in models_full:
            args.model = model
            print('model - {}'.format(model))
            file.write('\nModel: {}\n'.format(model))
            res = np.zeros((len(learning_rates), tests))
            for t in range(tests):
                embeddings, labels = embed_and_load(args)
                for i in range(len(learning_rates)):
                    lr = learning_rates[i]
                    train, train_labels, test, test_labels = select(embeddings, labels, train_frac, test_frac)
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
            plt.savefig('res/img/cora{}lr{}{}.png'.format(lr, model, get_time()))
            plt.show()

def loop_classify_reweightings(args, train_frac, test_frac, reweight_value):
    with open("./res/cora_rew{}{}.txt".format(reweight_value, get_time()), "w") as file:
        file.write(f"training set fraction is {train_frac}, test set fraction is {test_frac}, reweight value is {reweight_value}")
        for model in models:
            args.model = model
            print('model - {}'.format(model))
            file.write('\nModel: {}\n'.format(model))
            res = np.zeros((len(learning_rates), tests))
            for t in range(tests):
                for i in range(len(learning_rates)):
                    train, train_labels, test, test_labels = embed_and_select(args, train_frac, test_frac, reweight=True, reweight_value=reweight_value)
                    lr = learning_rates[i]
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
            plt.savefig('res/img/cora_rew{}lr{}trf{}{}.png'.format(lr, model, train_frac, get_time()))
            plt.show()

# def loop_classify_reweightings_fixed_lr(args, train_frac, test_frac, reweight_value, lr):
#     with open("./res/cora_rew{}{}.txt".format(reweight_value, time.time()), "w") as file:
#         file.write(f"training set fraction is {train_frac}, test set fraction is {test_frac}")
#         for model in models:
#             args.model = model
#             print('model - {}'.format(model))
#             file.write('\nModel: {}\n'.format(model))
#             res = np.zeros((len(learning_rates), tests))
#             for t in range(tests):
#                 for i in range(len(learning_rates)):
#                     train, train_labels, test, test_labels = embed_and_select(args, train_frac, test_frac, reweight=True, reweight_value=reweight_value)
#                     lr = learning_rates[i]
#                     res[i, t] = classify(train, train_labels, test, test_labels, args, iterations, lr)
            
#             exps = np.arange(tests)
#             for i in range(len(learning_rates)):
#                 row = res[i, :]
#                 lr = learning_rates[i]
#                 file.write(get_single_info(row, lr))
#                 file.write(str(row))
#                 plt.plot(exps, row, label='learning rate = {}'.format(lr))
            
#             plt.title('Cora with {}'.format(model))
#             plt.legend()
#             plt.savefig('res/img/cora_rew{}lr{}trf{}.png'.format(lr, model, train_frac))
#             plt.show()

def loop_embed(args):
    filename = args.input.split('/')[-1][:-4]
    with open("./res/{}{}.txt".format(filename, get_time()), "w") as file:
        for m in models:
            print('\n{}\n'.format(m))
            args.model = m
            modularities = []
            times = []
            for i in range(10):
                print('trial {}'.format(i + 1))
                t = time.time()
                mod = create_and_run_model(args) 
                times.append(time.time() - t)
                modularities.append(mod)
            file.write('\nModel: {} \n'.format(m))
            file.write('{}\n'.format(get_report(np.asarray(modularities))))
            file.write(str(modularities))
            file.write('\nTimes:\n')
            file.write(str(times))

if __name__ == "__main__":
    args = parameter_parser()
    
    # for i in range(len(graphs)):
    #     print('\n\n\nRunning embedding with RAW ricci weights on graph {}\n\n\n'.format(graphs[i]))
    #     args.input = graphs[i]
    #     args.ricci_weights = curvatures[i]
    #     loop_embed(args)
    # loop_classify(args, 0.05, 0.1)
    # loop_classify_reweightings(args, 0.05, 0.1, 0.25)
    loop_embed(args)

    