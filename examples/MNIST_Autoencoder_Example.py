import os
import sys

import numpy as np
from sklearn.datasets import fetch_mldata

import tensorflow as tf

from modelwrangler.corral.convolutional_autoencoder import ConvolutionalAutoencoder
from modelwrangler.dataset_managers import CategoricalDataManager

sys.path.append(os.path.pardir)

EXAMPLE_DIR = os.path.curdir
DATA_DIR = os.path.join(EXAMPLE_DIR, 'mnist_data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Download MNIST dataset

mnist_data = fetch_mldata('MNIST original', data_home=DATA_DIR)
image_data = mnist_data['data'].reshape(-1, 28, 28, order='F')[..., np.newaxis]
del mnist_data

train_data = image_data[::1000, :, :, :]
test_data = image_data[1::1000, :, :, :]

# Initialize the model

model_name = "mnist_ae_example"

cae_model = ConvolutionalAutoencoder(
    name=model_name,
    in_size=[28, 28, 1],
    num_epochs=10,
    encode_nodes=[64],
    encode_params={
        "activation": 'relu',
        "dropout_rate": None,
        "kernel": [5, 5],
        "strides": [3, 3],
        "pool_size": [2, 2]
    },
    bottleneck_dim=128,
    bottleneck_params={
        "activation": 'relu',
        "act_reg": {'l1': 0.1},
        "dropout_rate": 0.1,
    },
    decode_nodes=[64],
    decode_params={
        "activation": 'relu',
        "dropout_rate": None,
        "kernel": [5, 5],
    },
    output_params={
        "dropout_rate": None,
        "activation": None,
        "act_reg": 'relu'
    })


# Run training

pre_mse = cae_model.score(test_data, test_data)
cae_model.train(train_data, train_data)
post_mse = cae_model.score(test_data, test_data)

print("Pre-training MSE: {:.1f}".format(pre_mse))
print("Post-training MSE: {:.1f}".format(post_mse))


# You can load the file from disk!

print("Loading file from disk")

param_file = os.path.join(
    model_name,
    '{}-params.json'.format(model_name)
)

restored_model = ConvolutionalAutoencoder.load(param_file)
post_mse_restored_model = restored_model.score(train_data, train_data)

print("Post-training MSE for restored model: {:.1f}".format(post_mse_restored_model))
