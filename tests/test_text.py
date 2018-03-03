"""End to end testing on feedforward models
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from nltk.corpus import brown

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager

from model_wrangler.model.losses import accuracy

from model_wrangler.model.corral.text_classification import TextClassificationModel

from model_wrangler.model.tester import ModelTester


CONV_PARAMS = {
    'name': 'test_text',
    'path': './tests/test_text',
    'graph': {
        'num_inputs': 1,
        'pad_length': 256,
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


def make_testdata(out_dim=3, num_samples=100):
    """Make sample data from brown corpus"""

    X = []
    y = []

    for idx, para in enumerate(brown.paras()):
        X.append(' '.join(para[0])[:256])

        _tmpy = np.zeros((out_dim,))
        _tmpy[idx % out_dim] = 1.0
        y.append(_tmpy)

        if idx > num_samples:
            break

    return X, y


def test_text_ff():
    """Test dense feedforward model"""

    ff_model = ModelWrangler(TextClassificationModel, CONV_PARAMS)

    out_dim = CONV_PARAMS['graph']['out_sizes'][0]
    X, y = make_testdata(out_dim=out_dim)

    dm1 = DatasetManager([X], [y])
    dm2 = DatasetManager([X], [y])
    ff_model.add_data(dm1, dm2)

    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))
    ff_model.train()
    print("Loss: {}".format(ff_model.score([X], [y])))
    print("Acc'y: {}".format(ff_model.score([X], [y], score_func=accuracy)))


if __name__ == "__main__":

    #print("\n\nunit testing text convolutional model")
    #ModelTester(
    #    ModelWrangler(TextClassificationModel, CONV_PARAMS)
    #)

    print("\n\ne2e testing text convolutional model")
    test_text_ff()
