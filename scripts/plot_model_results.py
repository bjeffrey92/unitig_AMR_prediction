import glob
import pickle 
import torch
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GNN_model.utils import R_or_S, breakpoints, load_training_data, \
                            load_testing_data, load_adjacency_matrix, \
                            DataGenerator
from GNN_model.train import epoch_, load_data
from GNN_model.models import GCNPerNode #to load saved model


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


def plot_model_accuracy_by_category(model_file, data_dir):
    training_data, testing_data, adj = load_data(data_dir)
    training_metadata = pd.read_csv(os.path.join(data_dir, 
                                                'training_metadata.csv'))
    testing_metadata = pd.read_csv( os.path.join(data_dir, 
                                                'testing_metadata.csv'))
    model = torch.load(model_file)

    train_preds = epoch_(model, training_data, adj)
    test_preds = epoch_(model, testing_data, adj)
    
    def pred_accuracy_by_category(predictions, labels, metadata, column, 
                                training_metadata = None):
        '''
        If training metadata is None, assumes that metadata is the training_metadata
        '''
        diff = abs(predictions - labels)
        c = pd.Series([int(i) < 1 for i in diff])

        correct = metadata[c][column].value_counts()
        incorrect = metadata[~c][column].value_counts()

        correct_missing = [i for i in metadata[column].unique()     
                                if i not in correct.keys()]
        for i in correct_missing:
            correct[i] = 0
        incorrect_missing = [i for i in metadata[column].unique() 
                                if i not in incorrect.keys()]
        for i in incorrect_missing:
            incorrect[i] = 0

        correct.sort_index(inplace = True)
        incorrect.sort_index(inplace = True)
        correct.name = 'Correct'
        incorrect.name = 'Incorrect'

        df = pd.concat([correct, incorrect], axis = 1)

        if training_metadata is None:
            df['Sum'] = df.Correct + df.Incorrect
        else:
            #get total counts of relevant categories in the training data
            totals = training_metadata[
                training_metadata[column].isin(
                    correct.index.to_list())][column].value_counts()
            totals.name = 'Sum'
            df = df.merge(pd.DataFrame(totals), 
                            right_index=True, 
                            left_index=True)

        df['Percent_correct'] = df.Correct/(df.Correct + df.Incorrect) * 100

        return df

    train_country_df = pred_accuracy_by_category(train_preds, 
                                                training_data.labels, 
                                                training_metadata, 'Country')
    test_country_df = pred_accuracy_by_category(test_preds, 
                                                testing_data.labels, 
                                                testing_metadata, 'Country',
                                                training_metadata)

    train_family_df = pred_accuracy_by_category(train_preds, 
                                            training_data.labels, 
                                            training_metadata, 'Family')
    test_family_df = pred_accuracy_by_category(test_preds, 
                                                testing_data.labels, 
                                                testing_metadata, 'Family',
                                                training_metadata)

    regex = r'(?<=/).*?(?=_fit)'
    phenotype = re.findall(regex, model_file)[0]

    plt.clf()

    plt.scatter(train_country_df.Sum, train_country_df.Percent_correct)
    plt.title(f'{phenotype} Training Accuracy per Country')
    plt.xlabel('#isolates in train set')
    plt.ylabel('Percentage Accuracy')
    plt.savefig(f'{phenotype}_training_country_accuracy.png')

    plt.clf()

    plt.scatter(test_country_df.Sum, test_country_df.Percent_correct)
    plt.title(f'{phenotype} Testing Accuracy per Country')
    plt.xlabel('#isolates in train set')
    plt.ylabel('Percentage Accuracy')
    plt.savefig(f'{phenotype}_testing_country_accuracy.png')

    plt.clf()

    plt.scatter(train_family_df.Sum, train_family_df.Percent_correct)
    plt.title(f'{phenotype} Training Accuracy per Family')
    plt.xlabel('#isolates in train set')
    plt.ylabel('Percentage Accuracy')
    plt.savefig(f'{phenotype}_training_family_accuracy.png')

    plt.clf()

    plt.scatter(test_family_df.Sum, test_family_df.Percent_correct)
    plt.title(f'{phenotype} Testing Accuracy per Family')
    plt.xlabel('#isolates in train set')
    plt.ylabel('Percentage Accuracy')
    plt.savefig(f'{phenotype}_testing_family_accuracy.png')

    plt.clf()


if __name__ == '__main__':
    lasso_results_files = glob.glob('lasso_model/results/*pkl')
    GNN_results_files = glob.glob('best_model/*s.tsv')
    VNN_results_files = glob.glob('best_model/*VNN.tsv')

    regex = r'(?<=log2_).*?(?=_mic)'
    lasso_results = {re.findall(regex, i)[0]: parse_lasso_results(i)
                        for i in lasso_results_files}
    GNN_results = {re.findall(regex, i)[0]: parse_NN_results(i)
                        for i in GNN_results_files}
    VNN_results = {re.findall(regex, i)[0]: parse_NN_results(i)
                        for i in VNN_results_files}

    plot_results(lasso_results, 
                'Linear Model Test and Train Accuracy', 'lasso_accuracy.png')
    plot_results(GNN_results, 'GNN Test and Train Accuracy', 'GNN_accuracy.png')
    plot_results(VNN_results, 'VNN Test and Train Accuracy', 'VNN_accuracy.png')

    GNN_models = glob.glob('best_model/*GNN.pt')
    regex2 = r'(?<=/).*?(?=_fit)'
    data_dirs = [f'data/model_inputs/freq_5_95/{re.findall(regex2, i)[0]}/'
                        for i in GNN_models]
    for i in range(len(data_dirs)):
        plot_model_accuracy_by_category(GNN_models[i], data_dirs[i])