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
from model_wrangler.model.corral.convolutional_triplet import ConvolutionalTripletModel


EXAMPLE_DIR = os.path.curdir
DATA_DIR = os.path.join(EXAMPLE_DIR, 'mnist_data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


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
image_labels = np.array(list(mnist_data['target']))
del mnist_data

# limit to 1% of the data for this
image_data = image_data[::100, ...]
image_labels = image_labels[::100, ...]

# Set up dataset managers

data_train = DatasetManager([image_data], [image_labels])
data_test = DatasetManager([image_data], [image_labels])

# Setting up a model

CONV_PARAMS = {
    'name': 'mnist_triplet_example',
    'path': './mnist_triplet_example',
    'graph': {
        'in_sizes': [[28, 28, 1]],
        'hidden_params': [
            {
                'num_units': 16,
                'kernel': 3,
                'strides': 1,
                'pool_size': 1,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.01,
            },
        ],
        'num_targets': 1, 
    },
}

TRAIN_PARAMS = {
    'epoch_length': 50,
    'num_epochs': 5,
    'batch_size': 10,
    'interval': 1
}


# Set up a dataset manager for categorical data
model = ModelWrangler(ConvolutionalTripletModel, CONV_PARAMS)
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
        'label': labels
    })

    fig, ax = plt.subplots(1,1)
    for label, data in df_embed.groupby('label'):
        ax.scatter(data.x, data.y, s=0.2, c=plt.cm.Spectral(data.label / data.label.max()))
        for coord in zip(data.x, data.y):
            ax.annotate(label, coord)
    fig.show()
    plt.savefig(os.path.join(EXAMPLE_DIR, 'mnist_triplet_example.png'))

embeds = model.predict([image_data])
plot_embeddings(embeds, image_labels)