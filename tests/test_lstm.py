"""End to end testing on feedforward models"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np

import numpy as np
from scipy.stats import zscore

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import SequentialDatasetManager

from model_wrangler.model.corral.lstm import LstmModel
from model_wrangler.model.corral.text_lstm import TextLstmModel
from model_wrangler.model.tester import ModelTester

from nltk.corpus import brown


LSTM_PARAMS = {
    'name': 'test_lstm',
    'path': './tests/test_lstm',
    'graph': {
        'max_string_size': 128,
        'in_sizes': [[16, 1]],
        'recurr_params': [
            {
                'units': 5,
                'dropout': 0.1,        
            },
            {
                'units': 5,
                'dropout': 0.1,        
            }
        ],
        'out_sizes': [1],
    }
}


def make_numeric_testdata(series_length=256, n_samp=1000):
    """Make sample data for linear regression"""

    X = [
        np.cumsum(np.random.rand(n_samp, series_length, 1), axis=1)
    ]
    return X

def test_lstm_numeric():
    """Test dense feedforward model"""


    X = make_numeric_testdata()

    dm = SequentialDatasetManager(
        X,
        in_win_len=LSTM_PARAMS['graph']['in_sizes'][0][0],
        out_win_len=LSTM_PARAMS['graph']['out_sizes'][0],
        cache_size=128
    )

    lstm_model = ModelWrangler(LstmModel, LSTM_PARAMS)
    lstm_model.add_data(dm, dm)

    xy_test = next(dm.get_next_batch(batch_size=128))

    print("Loss: {}".format(lstm_model.score(*xy_test)))
    lstm_model.train()
    print("Loss: {}".format(lstm_model.score(*xy_test)))


def make_text_testdata(n_samp=1000):
    """Make sample data for linear regression"""

    X = [
        ' '.join(para[0])
        for idx, para in enumerate(brown.paras())
        if idx < n_samp
    ]
    return [X]

def test_lstm_text():
    """Test dense feedforward model"""


    X = make_text_testdata()

    dm = SequentialDatasetManager(
        X,
        in_win_len=LSTM_PARAMS['graph']['in_sizes'][0][0],
        out_win_len=LSTM_PARAMS['graph']['out_sizes'][0],
        cache_size=128
    )

    lstm_model = ModelWrangler(TextLstmModel, LSTM_PARAMS)
    lstm_model.add_data(dm, dm)

    xy_test = next(dm.get_next_batch(batch_size=128))

    print("Loss: {}".format(lstm_model.score(*xy_test)))
    lstm_model.train()
    print("Loss: {}".format(lstm_model.score(*xy_test)))

if __name__ == "__main__":
    """
    print("\n\n LSTM unit tests")
    ModelTester(
        ModelWrangler(LstmModel, LSTM_PARAMS)
    )
    """
    print("\n\ne2e testing numeric LSTM")
    test_lstm_numeric()

    print("\n\ne2e testing text LSTM")
    test_lstm_text()
