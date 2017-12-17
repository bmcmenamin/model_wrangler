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


def make_testdata(in_dim=10, out_dim=3, num_samples=100):
    """Make sample data from brown corpus"""

    X = []
    y = []

    tp = tops.TextProcessor()

    for idx, para in enumerate(brown.paras()):

        intlist = tp.string_to_ints(' '.join(para[0]), pad_len=in_dim)
        X.append(np.array(intlist))

        _tmpy = np.zeros((out_dim,))
        _tmpy[idx % out_dim] = 1.0
        y.append(_tmpy)

        if idx > num_samples:
            break

    X = np.vstack(X)
    y = np.vstack(y)
    return X, y


def test_text_ff(out_dim=3):
    """Test dense autoencodes"""

    text_model = ConvolutionalText(
        conv_nodes=[10, 10],
        dense_nodes=[2],
        out_size=out_dim)

    X, y = make_testdata(
        in_dim=text_model.params.in_size,
        out_dim=out_dim,
        num_samples=100*out_dim
    )

    print("Loss: {}".format(text_model.score(X, y)))
    print("Acc'y: {}".format(text_model.score(X, y, score_func=tops.accuracy)))
    text_model.train(X, y)
    print("Loss: {}".format(text_model.score(X, y)))
    print("Acc'y: {}".format(text_model.score(X, y, score_func=tops.accuracy)))


if __name__ == "__main__":

    print("\n\nunit testing text convolutional model")
    ModelTester(ConvolutionalText)

    print("\n\ne2e testing text convolutional model")
    test_text_ff(out_dim=3)
