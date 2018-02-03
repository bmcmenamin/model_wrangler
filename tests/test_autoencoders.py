"""End to end testing on autoencoders
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101

import numpy as np
from scipy.stats import zscore

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager

from model_wrangler.model.corral.dense_autoencoder import DenseAutoencoderModel
from model_wrangler.model.corral.convolutional_autoencoder import ConvolutionalAutoencoderModel

from model_wrangler.model.tester import ModelTester


DENSE_PARAMS = {
    'name': 'test_ae_dense',
    'path': './tests/test_ae_dense',
    'graph': {
        'in_sizes': [10],
        'encoding_params': [
            {
                'num_units': 4,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            },
            {
                'num_units': 4,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            }
        ],
        'embed_params': {
            'num_units': 5,
            'bias': True,
            'activation': 'relu'
        },
        'decoding_params': [
            {
                'num_units': 4,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            },
            {
                'num_units': 4,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            }
        ],
    }
}


CONV_PARAMS = {
    'name': 'test_ae_conv',
    'path': './tests/test_ae_conv',
    'graph': {
        'in_sizes': [[10, 1]],
        'encoding_params': [
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            },
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            }
        ],
        'embed_params': {
            'num_units': 5,
            'bias': True,
            'activation': 'relu'
        },
        'decoding_params': [
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
            },
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'pool_size': 2,
                'bias': True,
                'activation': 'relu',
            }
        ],
    }
}


def make_timeseries_testdata(in_dim=100, n_samp=1000):
    """Make sample data for autoencoder test regression"""

    signal = zscore(np.random.randn(n_samp, 3), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]
    return X


def test_dense_ae():
    """Test dense autoencoder"""

    X = make_timeseries_testdata(in_dim=DENSE_PARAMS['graph']['in_sizes'][0])
    ae_model = ModelWrangler(DenseAutoencoderModel, DENSE_PARAMS)

    dm1 = DatasetManager([X], [X])
    dm2 = DatasetManager([X], [X])
    ae_model.add_data(dm1, dm2)

    print('TF training:')
    print('\tpre-score: {}'.format(ae_model.score([X], [X])))
    ae_model.train()
    print('\tpost-score: {}'.format(ae_model.score([X], [X])))


def test_conv_ae():
    """Test convolutional autoencoder"""

    X = make_timeseries_testdata(in_dim=CONV_PARAMS['graph']['in_sizes'][0][0])
    X = X[:, :, np.newaxis]

    ae_model = ModelWrangler(ConvolutionalAutoencoderModel, CONV_PARAMS)

    dm1 = DatasetManager([X], [X])
    dm2 = DatasetManager([X], [X])
    ae_model.add_data(dm1, dm2)

    print('TF training:')
    print('\tpre-score: {}'.format(ae_model.score([X], [X])))
    ae_model.train()
    print('\tpost-score: {}'.format(ae_model.score([X], [X])))


if __name__ == "__main__":

    print('\n\nunit testing dense autoencoder')
    ModelTester(
        ModelWrangler(DenseAutoencoderModel, DENSE_PARAMS)
    )

    print("\n\ne2e testing dense autoencoder")
    test_dense_ae()

    print('\n\nunit testing convolutional autoencoder')
    ModelTester(
        ModelWrangler(ConvolutionalAutoencoderModel, CONV_PARAMS)
    )

    print("\n\ne2e testing convolutional autoencoder")
    test_conv_ae()
