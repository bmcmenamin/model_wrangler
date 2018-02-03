import os
import sys

import numpy as np
from sklearn.datasets import fetch_mldata


from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager
from model_wrangler.model.corral.convolutional_feedforward import ConvolutionalFeedforwardModel


from model_wrangler.model.losses import accuracy



sys.path.append(os.path.pardir)

EXAMPLE_DIR = os.path.curdir
DATA_DIR = os.path.join(EXAMPLE_DIR, 'mnist_data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Download MNIST dataset

def onehot_encoding(categories, max_categories):
    """Given a list of integer categories (out of a set of max_categories)
    return one-hot enocded values"""

    out_array = np.zeros((len(categories), max_categories))
    for key, val in enumerate(categories):
        out_array[key, int(val)] = 1.0

    return out_array


mnist_data = fetch_mldata('MNIST original', data_home=DATA_DIR)

image_data = mnist_data['data'].reshape(-1, 28, 28, order='F')[..., np.newaxis]
image_labels = onehot_encoding(list(mnist_data['target']), 10)
del mnist_data

train_data = image_data[::10, :, :, :]
train_labels = image_labels[::10, :]

test_data = image_data[1::10, :, :, :]
test_labels = image_labels[1::10, :]


data_train = DatasetManager([train_data], [train_labels])
data_test = DatasetManager([test_data], [test_labels])




#
# Create a model
#

CONV_PARAMS = {
    'name': 'mnist_clf_example',
    'path': './mnist_clf_example',
    'graph': {
        'in_sizes': [[28, 28, 1]],
        'hidden_params': [
            {
                'num_units': 32,
                'kernel': [5, 5],
                'strides': 2,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.1,
            },
            {
                'num_units': 128,
                'kernel': [5, 5],
                'strides': 2,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.1,
            },
        ],
        'embed_params': {
            'num_units': 10,
            'bias': True,
            'activation': 'relu',
            'dropout_rate': 0.0,
        },
        'out_sizes': [10], 
    }
}

TRAIN_PARAMS = {
    'num_epochs': 20,
    'batch_size': 32
}



# Set up a dataset manager for categorical data
model = ModelWrangler(ConvolutionalFeedforwardModel, CONV_PARAMS)
model.add_train_params(TRAIN_PARAMS)
model.add_data(data_train, data_test)


# Run training
pre_accy = 100 * model.score([test_data], [test_labels], score_func=accuracy)
print("Pre-training acc'y: {:.1f}%".format(pre_accy))

model.train()

post_accy = 100 * model.score([test_data], [test_labels], score_func=accuracy)
print("Post-training acc'y: {:.1f}%".format(post_accy))


# You can load the file from disk!

print("Loading file from disk")

param_file = os.path.join(CONV_PARAMS['path'], 'model_params.pickle')

restored_model = ModelWrangler.load(param_file)
post_accy_restored_model = 100 * restored_model.score([test_data], [test_labels], score_func=accuracy)
print("Post-training acc'y for restored model: {:.1f}%".format(post_accy_restored_model))
