import os
import sys

import numpy as np
from sklearn.datasets import fetch_mldata

from modelwrangler.corral.convolutional_feedforward import ConvolutionalFeedforward
from modelwrangler.dataset_managers import CategoricalDataManager

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

# Initialize the model

model_name = "mnist_clf_example"

conv_ff_network = ConvolutionalFeedforward(
    name=model_name,
    verb=True,
    in_size=[28, 28, 1],
    out_size=10,
    conv_nodes=[16, 24],
    conv_params={
        'dropout_rate': 0.0,
        'kernel': [5, 5],
        'strides': 2,
        'pool_size': 2,
    },
    dense_nodes=[10],
    dense_params={
        'dropout_rate': 0.1,
        'activation': 'relu',
        'act_reg': None,
    },
    output_params={
        'dropout_rate': None,
        'activation': 'softmax',
        'act_reg': None,
    },
    num_epochs=10
)

# Set up a dataset manager for categorical data
conv_ff_network.tf_mod.DATA_CLASS = CategoricalDataManager

# Run training

pre_accy = 100 * conv_ff_network.score(
    test_data, test_labels,
    modelwrangler.tf_ops.accuracy
)

conv_ff_network.train(train_data, train_labels)

post_accy = 100 * conv_ff_network.score(
    test_data, test_labels,
    modelwrangler.tf_ops.accuracy
)

print("Pre-training acc'y: {:.1f}%".format(pre_accy))
print("Post-training acc'y: {:.1f}%".format(post_accy))


# You can load the file from disk!

print("Loading file from disk")

param_file = os.path.join(
    model_name,
    '{}-params.json'.format(model_name)
)

restored_model = ConvolutionalFeedforward.load(param_file)

post_accy_restored_model = 100 * restored_model.score(
    test_data,
    test_labels,
    modelwrangler.tf_ops.accuracy
)

print("Post-training acc'y for restored model: {:.1f}%".format(post_accy_restored_model))
