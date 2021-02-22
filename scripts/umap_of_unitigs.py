import umap
import torch
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

from GNN_model.utils import R_or_S, breakpoints, load_metadata

def load_data(data_dir):
    train_features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    train_features = train_features.to_dense().tolist()
    train_labels = torch.load(os.path.join(data_dir, 
                                        'training_labels.pt')).tolist()

    test_features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    test_features = test_features.to_dense().tolist()
    test_labels = torch.load(os.path.join(data_dir, 
                                        'testing_labels.pt')).tolist()

    features = np.array(train_features + test_features)
    labels = np.array(train_labels + test_labels)

    metadata = pd.concat(list(load_metadata(data_dir)))
    
    return features, labels, metadata.Sanger_lane.to_numpy()


def main(Ab):
    data_dir = os.path.join(root_dir, Ab)

    pat = r'.*?\_(.*)_.*'
    ab_name = re.findall(pat, Ab)[0]

    features, labels, ids = load_data(data_dir)
    r_s_labels = R_or_S(labels, breakpoints[ab_name])

    reducer = umap.UMAP(metric = 'jaccard')
    embedding = reducer.fit_transform(features)

    embedding_df = pd.DataFrame(embedding)
    embedding_df['labels'] = labels
    embedding_df['r_s_labels'] = r_s_labels
    embedding_df['ids'] = ids
    embedding_df.to_csv(f'umap_results/jaccard_distance/{Ab}_umap_embedding.csv')

    plt.clf()
    plt.scatter(x = embedding_df[0], y = embedding_df[1], c = embedding_df.r_s_labels)
    plt.savefig(f'umap_results/jaccard_distance/{Ab}_umap_coloured_by_resistance_status.png')


if __name__ == '__main__':

    root_dir = 'data/model_inputs/freq_5_95'
    Abs = os.listdir(root_dir)

    for Ab in Abs:
        main(Ab)
