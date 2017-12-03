"""End to end testing on siamese nets
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from modelwrangler.corral.convolutional_siamese import ConvolutionalSiamese
from modelwrangler.tester import ModelTester


def make_timeseries_testdata(in_dim=100, n_samp=1000):
    """Make sample data for linear regression
    """

    signal = zscore(np.random.randn(n_samp, 3), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]
    return X

def test_conv_siamese(dim=48):
    """Test dense autoencodes
    """

    X0 = make_timeseries_testdata(in_dim=dim)
    X0 = X0[:, :, np.newaxis]

    X1 = make_timeseries_testdata(in_dim=dim)
    X1 = X1[:, :, np.newaxis]

    Y = np.array([i % 2 for i in range(X0.shape[0])]).reshape(-1, 1)

    convsiam_network = ConvolutionalSiamese(
        in_size=dim,
        out_size=3,
        conv_nodes=[3],
        conv_params={
            'dropout_rate': 0.1,
            'kernel': 3,
            'strides': 2,
            'pool_size': 2,
        },
        dense_nodes=[2],
        dense_params={
            'dropout_rate': 0.1,
            'activation': 'relu',
        },
        output_params={
            "dropout_rate": None,
            "activation": 'linear',
        },
    )


    print(convsiam_network.score([X0, X1], Y))
    for _ in range(5):
        convsiam_network.train([X0, X1], Y)
        print(convsiam_network.score([X0, X1], Y))

if __name__ == "__main__":

    print('\n\nunit testing siamese net')
    ModelTester(ConvolutionalSiamese)

    print("\n\ne2e testing ConvolutionalSiamese")
    test_conv_siamese()
