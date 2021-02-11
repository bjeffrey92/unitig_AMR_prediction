import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import os
import re


def load_data(data_dir):
    input_files = glob.glob(data_dir + 'log2*pkl')
    Abs = [os.path.split(i)[-1].split('_mic')[0] + '_mic' for i in input_files]
    data_dict = {}
    for i in range(len(Abs)):
        with open(input_files[i], 'rb') as a:
            data_dict[Abs[i]] = pickle.load(a)
    return data_dict


def convert_to_dataframe(CV_results):
    df_dictionary = {
            'left_out_clade' : [],
            'training_accuracy' : [],
            'testing_accuracy' : [],
            'validation_accuracy' : []
    } #will be converted to df
    for left_out_clade, d in CV_results.items():
        d_val = {i:d[i]['validation_accuracy'] for i in d.keys()}
        k = max(d_val, key = d_val.get) #key with max validation data accuracy
        df_dictionary['left_out_clade'].append(left_out_clade)
        df_dictionary['training_accuracy'].append(d[k]['training_accuracy'])
        df_dictionary['testing_accuracy'].append(d[k]['testing_accuracy'])
        df_dictionary['validation_accuracy'].append(d[k]['validation_accuracy'])
    
    return pd.DataFrame(df_dictionary)


def get_NN_results(data_dir, file_suffix = '.tsv'):
    '''
    Returns data from NN fitting metrics in form to be plotted with plot_results
    '''
    if not data_dir.endswith('/'):
        data_dir += '/'

    input_files = glob.glob(data_dir + f'*{file_suffix}')
    Abs = [os.path.split(i)[-1].split('_mic')[0] + '_mic' for i in input_files]
    
    data_dict = {}
    for Ab in set(Abs):
        Ab_files = [i for i in input_files if i.startswith(data_dir + Ab)]
        Ab_accuracies = [None] * len(Ab_files)
        i = 0
        for f in Ab_files:
            left_out_clade = re.search(r'clade_(.*?)_left', f).group(1)
            last_line = pd.read_csv(Ab_files[0], sep = '\t').iloc[-1]
            accuracies = last_line[['training_data_acc', 
                                    'testing_data_acc',
                                    'validation_data_acc']]
            accuracies['left_out_clade'] = int(left_out_clade)
            Ab_accuracies[i] = accuracies
            i += 1
        df = pd.concat(Ab_accuracies, axis = 1).transpose()
        df.set_index('left_out_clade', inplace = True)
        df.columns = ['training_accuracy', 
                    'testing_accuracy',
                    'validation_accuracy']
        data_dict[Ab] = df

    return data_dict


def plot_results(data, filename):
    fig, axs = plt.subplots(1, 4, sharey = True)
    width = 0.2
    n = 0
    for Ab in data.keys():
        df = data[Ab]
        axs[n].bar(df.index - width, df.training_accuracy, width, 
                label = 'Training Data')
        axs[n].bar(df.index, df.testing_accuracy, width, 
                label = 'Testing Data')
        axs[n].bar(df.index + width, df.validation_accuracy, width, 
                label = 'Validation Data')
        axs[n].set_xticks(df.index)
        axs[n].set_title(Ab.upper().split('_')[1])
        n += 1

    axs[n -1].legend(loc = 'lower right')
    fig.text(0.5, 0.04, 'Left Out Clade', ha='center')
    fig.text(0.04, 0.5, 'Prediction Accuracy (%)', 
            va='center', rotation='vertical')

    fig.savefig(filename)


if __name__ == '__main__':
    data_dir = 'lasso_model/results/linear_model_results/cross_validation_results/'
    data = load_data(data_dir)

    Abs = list(data.keys())
    Abs.sort() #to maintain order
    data = {Ab:convert_to_dataframe(data[Ab]) for Ab in Abs}
    data = {Ab:df.set_index('left_out_clade') for Ab, df in data.items()}

    plot_results(data, os.path.join(data_dir, 'CV_accuracy.png'))