"""
Module sets up Linear Regression model
"""

import tensorflow as tf

from model_wrangler.model_wrangler import ModelWrangler
import model_wrangler.tf_ops as tops
from model_wrangler.tf_models import BaseNetworkParams, BaseNetwork

class LinearRegressionParams(BaseNetworkParams):
    """Linear regression defaul params
    """

    MODEL_SPECIFIC_ATTRIBUTES = {
        'name': 'linreg',
        'in_size': 10,
        'out_size': 1,
    }

class LinearRegressionModel(BaseNetwork):
    """Linear regression model spec
    """

    # pylint: disable=too-many-instance-attributes

    PARAM_CLASS = LinearRegressionParams

    def setup_layers(self, params):
        """Build all the model layers
        """
        in_layer = tf.placeholder(
            "float",
            name="input",
            shape=[None, params.in_size]
            )

        coeff = tf.Variable(tf.ones([params.in_size, 1]), name="coeff")
        intercept = tf.Variable(tf.zeros([1,]), name="intercept")
        out_layer = tf.add(tf.matmul(in_layer, coeff), intercept, name="output")

        target_layer = tf.placeholder(
            "float",
            name="target",
            shape=[None, params.out_size]
        )

        loss = tops.loss_mse(target_layer, out_layer)

        return in_layer, out_layer, target_layer, loss

class LinearRegression(ModelWrangler):
    """Linear regression modelwrangle
    """

    def __init__(self, in_size=10, **kwargs):
        super(LinearRegression, self).__init__(
            model_class=LinearRegressionModel,
            in_size=in_size,
            **kwargs)
