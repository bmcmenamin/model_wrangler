"""End to end testing on simple models
"""

# pylint: disable=C0103
# pylint: disable=C0325
# pylint: disable=E1101


import numpy as np
from scipy.stats import zscore

from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.linear_model import LinearRegression as sk_LinearRegression

from model_wrangler.corral.linear_regression import LinearRegression
from model_wrangler.corral.logistic_regression import LogisticRegression


def make_linear_reg_testdata(in_dim=2, n_samp=1000):
    """Make sample data for linear regression
    """
    signal = zscore(np.random.randn(n_samp, 1), axis=0)

    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    X += 0.2 * signal
    X = zscore(X, axis=0)
    y = signal + 100
    return X, y

def make_linear_cls_testdata(in_dim=2, n_samp=1000):
    """Make sample data for linear regression
    """
    signal = zscore(np.random.randn(n_samp, 1), axis=0)
    X = zscore(np.random.randn(n_samp, in_dim), axis=0)
    X += 0.2 * signal
    X = zscore(X, axis=0)
    y = (signal > 0).astype(int)
    return X, y


def compare_scikt_and_tf(sk_class, tf_class, X, y, sk_params={}):

    sk_model = sk_class(**sk_params).fit(X, y.ravel())
    print('Scikit values:')
    print('\t coef: {}'.format(sk_model.coef_.ravel()))
    print('\t int: {}'.format(sk_model.intercept_.ravel()))


    tf_model = tf_class(in_size=X.shape[1], num_epochs=50)

    print('TF training:')
    print('\tpre-score: {}'.format(tf_model.score(X, y)))
    tf_model.train(X, y)
    print('\tpost-score: {}'.format(tf_model.score(X, y)))

    print('TF values:')
    print('\t coef: {}'.format(tf_model.get_from_model('coeff').ravel()))
    print('\t int: {}'.format(tf_model.get_from_model('intercept').ravel()))

    try:
        corr = np.mean(
            zscore(tf_model.predict(X).ravel()) *
            zscore(sk_model.predict_proba(X)[:, 1].ravel())
        )
    except AttributeError:
        corr = np.mean(
            zscore(tf_model.predict(X).ravel()) *
            zscore(sk_model.predict(X).ravel())
        )

    print('Model Correlation')
    print('\tr = {:.2f}'.format(corr))


def test_linear_regr(in_dim=2):
    """Compare tf linear regression to scikit learn
    """
    X, y = make_linear_reg_testdata(in_dim=in_dim)

    compare_scikt_and_tf(
        sk_LinearRegression,
        LinearRegression,
        X, y)


def test_logistic_regr(in_dim=2):
    """Compare tf logistic regression to scikit learn
    """
    X, y = make_linear_cls_testdata(in_dim=in_dim)

    compare_scikt_and_tf(
        sk_LogisticRegression,
        LogisticRegression,
        X, y,
        sk_params={'penalty':'l2', 'C':100.0})


if __name__ == "__main__":

    print("testing linear regression")
    test_linear_regr()

    print("testing logistic regression")
    test_logistic_regr()
