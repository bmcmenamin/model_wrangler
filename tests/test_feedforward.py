"""End to end testing on feedforward models"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager


from model_wrangler.model.losses import accuracy

from model_wrangler.model.corral.dense_feedforward import DenseFeedforwardModel
from model_wrangler.model.corral.convolutional_feedforward import ConvolutionalFeedforwardModel

from model_wrangler.model.tester import ModelTester


DENSE_PARAMS = {
    'name': 'test_ff_dense',
    'path': './tests/test_ff_dense',
    'graph': {
        'in_sizes': [10],
        'hidden_params': [
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
        'out_sizes': [5], 
    }
}


CONV_PARAMS = {
    'name': 'test_ff_conv',
    'path': './tests/test_ff_conv',
    'graph': {
        'in_sizes': [[10, 1]],
        'hidden_params': [
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.0,
            },
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
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
        'out_sizes': [5], 
    }
}

def make_testdata(in_dim=100, out_dim=3, n_samp=1000):
    """Make sample data for linear regression"""

    signal = zscore(np.random.randn(n_samp, out_dim), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]

    y = signal == np.max(signal, axis=1, keepdims=True)
    return X, y


def test_dense_ff():
    """Test dense feedforward model"""

    ff_model = ModelWrangler(DenseFeedforwardModel, DENSE_PARAMS)

    in_dim = DENSE_PARAMS['graph']['in_sizes'][0]
    out_dim = DENSE_PARAMS['graph']['out_sizes'][0]
    X, y = make_testdata(in_dim=in_dim, out_dim=out_dim)

    dm1 = DatasetManager([X], [y])
    dm2 = DatasetManager([X], [y])
    ff_model.add_data(dm1, dm2)

    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))
    ff_model.train()
    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))


def test_conv_ff(in_dim=15, out_dim=3):
    """Test dense feedforward"""

    ff_model = ModelWrangler(ConvolutionalFeedforwardModel, CONV_PARAMS)

    in_dim = CONV_PARAMS['graph']['in_sizes'][0][0]
    out_dim = CONV_PARAMS['graph']['out_sizes'][0]
    X, y = make_testdata(in_dim=in_dim, out_dim=out_dim)
    X = X[..., np.newaxis]

    dm1 = DatasetManager([X], [y])
    dm2 = DatasetManager([X], [y])
    ff_model.add_data(dm1, dm2)

    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))
    ff_model.train()
    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))


if __name__ == "__main__":

    print("\n\nunit testing dense feedforward")
    ModelTester(
        ModelWrangler(DenseFeedforwardModel, DENSE_PARAMS)
    )

    print("\n\ne2e testing dense feedforward")
    test_dense_ff()

    print("\n\nunit testing conv feedforward")
    ModelTester(
        ModelWrangler(ConvolutionalFeedforwardModel, CONV_PARAMS)
    )

    print("\n\ne2e testing conv feedforward")
    test_conv_ff()
