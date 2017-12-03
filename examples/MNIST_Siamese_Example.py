import os
import sys

from itertools import combinations

import numpy as np
from sklearn.datasets import fetch_mldata

import tensorflow as tf
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from modelwrangler.corral.convolutional_siamese import ConvolutionalSiamese

sys.path.append(os.path.pardir)

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


def make_training_pairs(x, y):
    input_0 = []
    input_1 = []
    output = []
    for i in combinations(range(x.shape[0]), r=2):
        input_0.append(x[i[0]: (i[0] + 1), ...])
        input_1.append(x[i[1]: (i[1] + 1), ...])
        output.append(np.sum(np.prod(y[i, ...], axis=0)))

    input_0 = np.vstack(input_0)
    input_1 = np.vstack(input_1)
    output = np.vstack(output)
    return input_0, input_1, output


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


# Download MNIST dataset

mnist_data = fetch_mldata('MNIST original', data_home=DATA_DIR)

image_data = mnist_data['data'].reshape(-1, 28, 28, order='F')[..., np.newaxis]
image_labels = onehot_encoding(list(mnist_data['target']), 10)
del mnist_data

# Set up training data

subsample_factor = 100
train_data = image_data[::subsample_factor, :, :, :]
train_labels = image_labels[::subsample_factor, :]
train_pairs0, train_pairs1, train_pairs_labels = make_training_pairs(train_data, train_labels)

# Setting up a model

model_name = "siamese_example"

convsiam_network = ConvolutionalSiamese(
    name=model_name,
    verb=True,
    in_size=[28, 28, 1],
    out_size=3,
    conv_nodes=[16, 24, 8],
    conv_params={
        'dropout_rate': 0.1,
        'kernel': [5, 5],
        'strides': 2,
        'pool_size': 2,
    },
    dense_nodes=[10, 5],
    dense_params={
        'dropout_rate': 0.1,
        'activation': 'relu',
    },
    output_params={
        "dropout_rate": None,
        "activation": 'linear',
    },
    num_epochs=5
)

# Train it to learn that pictures of the same number should be next to each other
convsiam_network.train([train_pairs0, train_pairs1], train_pairs_labels, pos_classes=[(1.0,)])

# Get the embedding vectors for each input, and plot the output to screen and disk
train_embeddings = convsiam_network.get_embedding_score(train_data)
plot_embeddings(train_embeddings[::5], train_labels[::5])
