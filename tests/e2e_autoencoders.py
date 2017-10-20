"""End to end testing on autoencoders
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from model_wrangler.corral.dense_autoencoder import DenseAutoencoder

from model_wrangler.tf_models import ConvLayerConfig, LayerConfig
from model_wrangler.corral.convolutional_autoencoder import ConvolutionalAutoencoder


def make_timeseries_testdata(in_dim=100, n_samp=1000):
    """Make sample data for linear regression
    """

    signal = zscore(np.random.randn(n_samp, 3), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]
    return X


def test_dense_ae(dim=24):
    """Test dense autoencodes
    """

    X = make_timeseries_testdata(in_dim=dim)
    ae_model = DenseAutoencoder(in_size=dim)

    print(ae_model.score(X, X))
    ae_model.train(X, X)
    print(ae_model.score(X, X))

def test_conv_ae(dim=48):
    """Test dense autoencodes
    """

    X = make_timeseries_testdata(in_dim=dim)
    X = X[:, :, np.newaxis]

    cae_model = ConvolutionalAutoencoder(
        in_size=dim,
        encode_nodes=[3],
        encode_params={
            "activation": None,
            "dropout_rate": None,
            "kernel": 3,
            "strides": 1,
            "pool_size": 3
        },
        bottleneck_dim=2,
        bottleneck_params={
            "activation": None,
            "dropout_rate": None,
            "kernel": 3,
            "strides": 1,
            "pool_size": 3
        },
        decode_nodes=[3],
        decode_params={
            "activation": None,
            "dropout_rate": None,
            "kernel": 3,
            "strides": 1,
            "pool_size": 3
        },
        output_params={
            "dropout_rate": None,
            "activation": None,
            "act_reg": None,
            "kernel": 1,
            "strides": 1,
            "pool_size": 1
        })

    print(cae_model.score(X, X))
    cae_model.train(X, X)
    print(cae_model.score(X, X))

if __name__ == "__main__":

    #print("\n\ntesting dense autoencoder")
    #test_dense_ae()

    print("\n\ntesting convolutional autoencoder")
    test_conv_ae()
