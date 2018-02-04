import os
import sys

import numpy as np
from sklearn.datasets import fetch_mldata

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager
from model_wrangler.model.corral.convolutional_autoencoder import ConvolutionalAutoencoderModel


sys.path.append(os.path.pardir)

EXAMPLE_DIR = os.path.curdir
DATA_DIR = os.path.join(EXAMPLE_DIR, 'mnist_data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Download MNIST dataset

mnist_data = fetch_mldata('MNIST original', data_home=DATA_DIR)

image_data = mnist_data['data'].reshape(-1, 28, 28,  order='F')[..., np.newaxis]
del mnist_data

train_data = image_data[::2, :, :, :]
test_data = image_data[1::2, :, :, :]

data_train = DatasetManager([train_data], [train_data])
data_test = DatasetManager([test_data], [test_data])


#
# Create a model
#

CONV_PARAMS = {
    'name': 'mnist_ae_example',
    'path': './mnist_ae_example',
    'graph': {
        'in_sizes': [[28, 28, 1]],
        'encoding_params': [
            {
                'num_units': 16,
                'kernel': [3, 3],
                'strides': 2,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0
            },
            {
                'num_units': 32,
                'kernel': [3, 3],
                'strides': 2,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0
            },        ],
        'embed_params': {
            'num_units': 32,
            'bias': True,
            'activation': 'relu'
        },
        'decoding_params': [
            {
                'num_units': 32,
                'kernel': [5, 5],
                'strides': 1,
                'pool_size': 1,
                'bias': True,
                'activation': 'relu',
                'dropout_rate': 0.0
            },
            {
                'num_units': 32,
                'kernel': [5, 5],
                'strides': 1,
                'pool_size': 1,
                'bias': True,
                'activation': 'relu',
                'dropout_rate': 0.0
            },
        ],
    }
}

TRAIN_PARAMS = {
    'num_epochs': 20,
    'batch_size': 64,
    'interval': 5
}


# Set up a dataset manager for categorical data
model = ModelWrangler(ConvolutionalAutoencoderModel, CONV_PARAMS)
model.add_train_params(TRAIN_PARAMS)
model.add_data(data_train, data_test)


# Run training
pre_mse = model.score([test_data], [test_data])
model.train()
post_mse = model.score([test_data], [test_data])

print("Pre-training mse: {:.3f}".format(pre_mse))
print("Post-training mse: {:.3f}".format(post_mse))

# You can load the file from disk!

print("Loading file from disk")

param_file = os.path.join(CONV_PARAMS['path'], 'model_params.pickle')

restored_model = ModelWrangler.load(param_file)
post_accy_restored_mse = restored_model.score([test_data], [test_data])
print("Post-training mse for restored model: {:.3f}".format(post_accy_restored_mse))