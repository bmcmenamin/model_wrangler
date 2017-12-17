"""End to end testing on feedforward models
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from nltk.corpus import brown

import modelwrangler.tf_ops as tops
from modelwrangler.tester import ModelTester

from modelwrangler.corral.convolutional_text_classification import ConvolutionalText


def make_testdata(num_samples=100, out_dim=3):
    """Make sample data from brown corpus
    """

    X = []
    y = []
    for idx, para in enumerate(brown.paras()):
        X.append(' '.join(para))
        _tmpy = np.zeros(out_dim,)
        _tmpy[idx % out_dim] = 1.0
        y.append(_tmpy)

        if idx > num_samples:
            break

    return X, y


def test_text_ff():
    """Test dense autoencodes
    """

    X, y = make_testdata(num_samples=100*out_dim, out_dim=out_dim)
    text_model = ConvolutionalText(
        in_size=in_dim,
        conv_nodes=[10, 10],
        dense_nodes=[2],
        out_size=out_dim)

    print("Loss: {}".format(text_model.score(X, y)))
    print("Acc'y: {}".format(text_model.score(X, y, score_func=tops.accuracy)))
    text_model.train(X, y)
    print("Loss: {}".format(text_model.score(X, y)))
    print("Acc'y: {}".format(text_model.score(X, y, score_func=tops.accuracy)))


if __name__ == "__main__":

    print("\n\nunit testing text convolutional model")
    ModelTester(ConvolutionalText)

    print("\n\ne2e testing text convolutional model")
    test_text_ff()

