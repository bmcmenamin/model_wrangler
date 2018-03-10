"""End to end testing on siamese nets
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager

from model_wrangler.model.losses import accuracy

from model_wrangler.model.corral.convolutional_siamese import ConvolutionalSiameseModel
from model_wrangler.model.corral.convolutional_triplet import ConvolutionalTripletModel
from model_wrangler.model.tester import ModelTester


CONV_PARAMS = {
    'name': '{}',
    'path': './tests/{}',
    'graph': {
        'in_sizes': [[100, 1]],
        'hidden_params': [
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.1,
            },
            {
                'num_units': 4,
                'kernel': 3,
                'strides': 1,
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
                'dropout_rate': 0.1,
            }
        ],
        'num_targets': 1, 
    },
}

TRAIN_PARAMS = {
    'num_epochs': 5,
    'batch_size': 10,
    'learning_rate': 0.001
}

def make_timeseries_testdata(in_dim=100, n_samp=1000):
    """Make sample data"""

    signal = zscore(np.random.randn(n_samp, 3), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    for i in range(X.shape[1]):
        X[:, i] += 0.1 * signal[:, (i % signal.shape[1])]
    return X

def test_conv_metric(name, model, n_samp=1000):
    """Test convolutional siamese network"""

    params = CONV_PARAMS.copy()
    params['name'] = name
    params['path'] = './tests/{}'.format(name)

    in_dim = params['graph']['in_sizes'][0][0]

    X = make_timeseries_testdata(in_dim=in_dim, n_samp=n_samp)
    X = X[..., np.newaxis]

    Y = np.array([i % 2 for i in range(n_samp)]).reshape(-1, 1)

    ff_model = ModelWrangler(model, params)
    ff_model.add_train_params(TRAIN_PARAMS)

    dm1 = DatasetManager([X], [Y])
    dm2 = DatasetManager([X], [Y])
    ff_model.add_data(dm1, dm2)

    print("Loss: {}".format(ff_model.score([X], [Y])))
    ff_model.train()
    print("Loss: {}".format(ff_model.score([X], [Y])))


if __name__ == "__main__":

    #print('\n\nunit testing siamese net')
    #ModelTester(ConvolutionalSiamese)

    print("\n\ne2e testing Convolutional Siamese")
    test_conv_metric('siam', ConvolutionalSiameseModel)

    print("\n\ne2e testing Convolutional Triplet")
    test_conv_metric('triplet', ConvolutionalTripletModel)
