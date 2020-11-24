import pickle 
import torch
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GNN_model.utils import R_or_S, breakpoints

def parse_lasso_results(lasso_results):
    with open(lasso_results, 'rb') as a:
        accuracy_dict = pickle.load(a)

    testing_results =  [accuracy_dict[i][1] for i in accuracy_dict.keys()]
    index = testing_results.index(max(testing_results))
    testing_acc = testing_results[index]
    training_acc = [accuracy_dict[i][0] for i in accuracy_dict.keys()][index]

    return training_acc, testing_acc


def parse_NN_results(results_file, delimiter = '\t'):
    df = pd.read_csv(results_file, delimiter = delimiter)
    last_line = df.iloc[-1]
    return last_line.training_data_acc, last_line.testing_data_acc


def plot_results(results_dict, title, fname):
    labels =  list(results_dict.keys())
    training_acc = [v[0] for _,v in results_dict.items()]
    testing_acc = [v[1] for _,v in results_dict.items()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training_acc, width, label = 'Training Set')
    rects2 = ax.bar(x + width/2, testing_acc, width, label = 'Testing Set')

    ax.set_ylabel('Percentage Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig(fname)

if __name__ == '__main__':
    lasso_results_files = glob.glob('lasso_model/results/*pkl')
    NN_results_files = []

    regex = r'(?<=log2_).*?(?=_mic)'
    lasso_results = {re.findall(regex, i)[0]: parse_lasso_results(i)
                        for i in lasso_results_files}
    NN_results = {re.findall(regex, i): parse_NN_results[i]
                        for i in NN_results_files}

