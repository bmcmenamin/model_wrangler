import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager
from model_wrangler.model.corral.convolutional_siamese import ConvolutionalSiameseModel


EXAMPLE_DIR = os.path.curdir
DATA_DIR = os.path.join(EXAMPLE_DIR, 'mnist_data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def onehot_encoding(categories, max_categories):
    """Given a list of integer categories (out of a set of max_categories)
    return one-hot enocded values"""

    out_array = np.zeros((len(categories), max_categories))
    for key, val in enumerate(categories):
        out_array[key, int(val)] = 1.0

    return out_array


def make_data(x, row_list):
    for i in row_list:
        yield x[i, ...]

def make_labels(y, pair_list):
    for pair in pair_list:
        output = np.sum(np.prod(y[pair, ...], axis=0))
        yield [output]

# Download MNIST dataset

mnist_data = fetch_mldata('MNIST original', data_home=DATA_DIR)

image_data = 1.0 * mnist_data['data'].reshape(-1, 28, 28, order='F')[..., np.newaxis]
image_labels = onehot_encoding(list(mnist_data['target']), 10)
del mnist_data

image_data = image_data[::100, ...]
image_labels = image_labels[::100, ...]

# Set up training data

pair_ordering = [i for i in combinations(range(image_data.shape[0]), r=2)]
random.shuffle(pair_ordering)

data_train = DatasetManager(
    [
        make_data(image_data, [i[0] for i in pair_ordering[:200]]),
        make_data(image_data, [i[1] for i in pair_ordering[:200]]),
    ],
    [make_labels(image_labels, pair_ordering[:200])]
)

data_test = DatasetManager(
    [
        make_data(image_data, [i[0] for i in pair_ordering[200:300]]),
        make_data(image_data, [i[1] for i in pair_ordering[200:300]]),
    ],
    [make_labels(image_labels, pair_ordering[200:300])]
)

# Setting up a model

CONV_PARAMS = {
    'name': 'mnist_siamese_example',
    'path': './mnist_siamese_example',
    'graph': {
        'in_sizes': [[28, 28, 1], [28, 28, 1]],
        'hidden_params': [
            {
                'num_units': 16,
                'kernel': 3,
                'strides': 1,
                'pool_size': 1,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            },
        ],
        'num_out': 1, 
    },
}

TRAIN_PARAMS = {
    'epoch_length': 20,
    'num_epochs': 5,
    'batch_size': 32,
    'interval': 1
}


# Set up a dataset manager for categorical data
model = ModelWrangler(ConvolutionalSiameseModel, CONV_PARAMS)
model.add_train_params(TRAIN_PARAMS)
model.add_data(data_train, data_test)

# Run training
model.train()



import matplotlib.pyplot as plt


def plot_embeddings(embed_data, labels):
    embed_twodim = TSNE().fit_transform(embed_data)

    df_embed = pd.DataFrame({
        'x': embed_twodim[:, 0],
        'y': embed_twodim[:, 1],
        'label': np.argmax(labels, axis=1)
    })

    fig, ax = plt.subplots(1,1)
    for label, data in df_embed.groupby('label'):
        ax.scatter(data.x, data.y, s=0.2, c=plt.cm.Spectral(label / labels.shape[1]))
        for coord in zip(data.x, data.y):
            ax.annotate(label, coord)
    fig.show()
    plt.savefig(os.path.join(EXAMPLE_DIR,'mnist_siamese_example.png'))

plot_embeddings(train_embeddings[::5], train_labels[::5])
