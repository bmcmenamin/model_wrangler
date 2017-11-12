"""End to end testing on feedforward models
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

import modelwrangler.tf_ops as tops
from modelwrangler.tester import ModelTester

from modelwrangler.tf_models import ConvLayerConfig, LayerConfig

from modelwrangler.corral.dense_feedforward import DenseFeedforward
from modelwrangler.corral.convolutional_feedforward import ConvolutionalFeedforward


def make_testdata(in_dim=100, out_dim=3, n_samp=1000):
    """Make sample data for linear regression
    """

    signal = zscore(np.random.randn(n_samp, out_dim), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]

    y = signal == np.max(signal, axis=1, keepdims=True)
    return X, y


def test_dense_ff(in_dim=15, out_dim=3):
    """Test dense autoencodes
    """

    X, y = make_testdata(in_dim=in_dim, out_dim=out_dim)
    ff_model = DenseFeedforward(
        in_size=in_dim,
        hidden_nodes=[2, 2],
        out_size=out_dim)

    print("Loss: {}".format(ff_model.score(X, y)))
    print("Acc'y: {}".format(ff_model.score(X, y, score_func=tops.accuracy)))
    ff_model.train(X, y)
    print("Loss: {}".format(ff_model.score(X, y)))
    print("Acc'y: {}".format(ff_model.score(X, y, score_func=tops.accuracy)))


def test_conv_ff(in_dim=15, out_dim=3):
    """Test dense autoencodes
    """

    X, y = make_testdata(in_dim=in_dim, out_dim=out_dim)
    X = X[:, :, np.newaxis]
    ff_model = ConvolutionalFeedforward(
        in_size=in_dim,
        hidden_nodes=[2, 2],
        out_size=out_dim)

    print("Loss: {}".format(ff_model.score(X, y)))
    print("Acc'y: {}".format(ff_model.score(X, y, score_func=tops.accuracy)))
    ff_model.train(X, y)
    print("Loss: {}".format(ff_model.score(X, y)))
    print("Acc'y: {}".format(ff_model.score(X, y, score_func=tops.accuracy)))


if __name__ == "__main__":

    print("\n\nunit testing dense feedforward")
    ModelTester(DenseFeedforward)

    print("\n\ne2e testing dense feedforward")
    test_dense_ff()

    print("\n\nunit testing conv feedforward")
    ModelTester(ConvolutionalFeedforward)

    print("\n\ne2e testing conv feedforward")
    test_conv_ff()
