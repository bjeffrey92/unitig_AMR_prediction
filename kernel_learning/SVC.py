import os 
import pickle 
import numpy as np

from lasso_model.utils import load_training_data, load_testing_data
from GNN_model.utils import R_or_S, breakpoints

from sklearn.svm import SVC

def fit_model(k_train, training_labels, k_test, testing_labels, Cs):
    summary_dict = {c: [None, None] for c in Cs}
    for C in Cs:
        model = SVC(C = C, kernel = 'precomputed').fit(k_train, training_labels)
        train_acc = model.score(k_train, training_labels) * 100 
        test_acc = model.score(k_test, testing_labels) * 100
        summary_dict[C] = [train_acc, test_acc]
        print(f'{C}\n', 
            f'Training Data Accuracy = {train_acc}\n',
            f'Testing Data Accuracy = {test_acc}\n')

    return summary_dict


if __name__ == '__main__':
    root_dir = 'data/model_inputs/freq_5_95/'
    Ab = 'log2_azm_mic'
    data_dir = os.path.join(root_dir, Ab)

    training_labels = load_training_data(data_dir)[1]
    testing_labels = load_testing_data(data_dir)[1]

    training_labels = R_or_S(training_labels.tolist(), 
                            breakpoints[Ab.split('_')[1]])
    testing_labels = R_or_S(testing_labels.tolist(), 
                            breakpoints[Ab.split('_')[1]])

    with open(f'kernel_learning/{Ab}/k_train.pkl', 'rb') as a:
        k_train = pickle.load(a)
    with open(f'kernel_learning/{Ab}/k_test.pkl', 'rb') as a:
        k_test = pickle.load(a)

    Cs =  np.linspace(1e-7, 1e-5, 20)
    summary_dict = fit_model(k_train, training_labels,
                            k_test, testing_labels, Cs)    