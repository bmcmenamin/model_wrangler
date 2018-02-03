"""End to end testing on simple linear models"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.linear_model import LinearRegression as sk_LinearRegression

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.dataset_managers import DatasetManager

from model_wrangler.model.corral.linear_regression import LinearRegressionModel
from model_wrangler.model.corral.logistic_regression import LogisticRegressionModel

from model_wrangler.model.tester import ModelTester


LINEAR_PARAMS = {
    'name': 'test_lin',
    'path': './tests/test_lin',
    'graph': {
        'in_sizes': [12],
        'out_sizes': [1], 
    }
}


LOGISTIC_PARAMS = {
    'name': 'test_log',
    'path': './tests/test_log',
    'graph': {
        'in_sizes': [12],
        'out_sizes': [1], 
    }
}

def make_linear_reg_testdata(in_dim=2, n_samp=1000):
    """Make sample data for linear regression"""

    signal = zscore(np.random.randn(n_samp, 1), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    X += 0.2 * signal
    X = zscore(X, axis=0)
    y = signal + 100
    return X, y

def make_linear_cls_testdata(in_dim=2, n_samp=1000):
    """Make sample data for logistic regression"""

    signal = zscore(np.random.randn(n_samp, 1), axis=0)
    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    X += 0.2 * signal
    X = zscore(X, axis=0)
    y = (signal > 0).astype(int)
    return X, y


def compare_scikt_and_tf(sk_model, tf_model, X, y, sk_params={}):

    sk_model = sk_model.fit(X, y.ravel())
    print('Scikit values:')
    print('\t coef: {}'.format(sk_model.coef_.ravel()))
    print('\t int: {}'.format(sk_model.intercept_.ravel()))

    dm1 = DatasetManager([X], [y])
    dm2 = DatasetManager([X], [y])
    tf_model.add_data(dm1, dm2)

    print('TF training:')
    print('\tpre-score: {}'.format(tf_model.score([X], [y])))
    tf_model.train()
    print('\tpost-score: {}'.format(tf_model.score([X], [y])))

    print('TF values:')
    print('\t coef: {}'.format(tf_model.get_from_model('params/coeff_0').ravel()))
    print('\t int: {}'.format(tf_model.get_from_model('params/intercept_0').ravel()))

    try:
        corr = np.mean(
            zscore(tf_model.predict([X])[0].ravel()) *
            zscore(sk_model.predict_proba(X)[:, 1].ravel())
        )
    except AttributeError:
        corr = np.mean(
            zscore(tf_model.predict([X])[0].ravel()) *
            zscore(sk_model.predict(X).ravel())
        )

    print('Model Correlation')
    print('\tr = {:.2f}'.format(corr))


def test_linear_regr():
    """Compare tf linear regression to scikit learn"""

    X, y = make_linear_reg_testdata(in_dim=LINEAR_PARAMS['graph']['in_sizes'][0])

    compare_scikt_and_tf(
        sk_LinearRegression(),
        ModelWrangler(LinearRegressionModel, LINEAR_PARAMS),
        X, y)

def test_logistic_regr():
    """Compare tf logistic regression to scikit learn"""

    X, y = make_linear_cls_testdata(in_dim=LOGISTIC_PARAMS['graph']['in_sizes'][0])

    compare_scikt_and_tf(
        sk_LogisticRegression(**{'penalty':'l2', 'C':100.0}),
        ModelWrangler(LogisticRegressionModel, LOGISTIC_PARAMS),
        X, y)

if __name__ == "__main__":

    print("\n\nunit testing linear regression")
    ModelTester(
        ModelWrangler(LinearRegressionModel, LINEAR_PARAMS)
    )

    print("\n\ne2e testing linear regression")
    test_linear_regr()

    print("\n\nunit testing logistic regression")
    ModelTester(
        ModelWrangler(LogisticRegressionModel, LOGISTIC_PARAMS)
    )

    print("\n\ne2e testing logistic regression")
    test_logistic_regr()
