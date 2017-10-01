"""
Module sets up Linear Regression model
"""
import tensorflow as tf

import tf_ops as tops
from tf_models import BaseNetworkParams, BaseNetwork
from model_wrangler import ModelWrangler

import dataset_managers as dm

class LogisticRegressionParams(BaseNetworkParams):
    """Linear regression defaul params
    """
    MODEL_SPECIFIC_ATTRIBUTES = {
        'in_size': 10,
        'out_size': 1,
    }

class LogisticRegressionModel(BaseNetwork):
    """Linear regression model spec
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = LogisticRegressionParams
    DATA_CLASS = dm.CategoricalDataManager

    def setup_layers(self, params):
        """Build all the model layers
        """
        in_layer = tf.placeholder(
            "float",
            name="input",
            shape=[None, params.in_size]
            )

        coeff = tf.Variable(tf.ones([params.in_size, 1]), name="coeff")
        intercept = tf.Variable(tf.ones([1,]), name="intercept")
        out_layer = tf.sigmoid(
            tf.add(tf.matmul(in_layer, coeff), intercept),
            name='output')

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.out_size]
        )

        loss = tops.loss_sigmoid_ce(target_layer, out_layer)

        return in_layer, out_layer, target_layer, loss

class LogisticRegression(ModelWrangler):
    """Linear regression modelwrangle
    """

    def __init__(self, in_size=10, **kwargs):
        super(LogisticRegression, self).__init__(
            model_class=LogisticRegressionModel,
            in_size=in_size,
            **kwargs)
