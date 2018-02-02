"""Module sets up Linear Regression model"""

# pylint: disable=R0914

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.losses import loss_mse


class LinearRegressionModel(BaseArchitecture):
    """Linear regression"""

    def setup_layers(self, params):
        """Build all the model layers"""

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        out_sizes = params.get('out_sizes', [])

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("float", name="input_{}".format(idx), shape=[None, in_size])
            for idx, in_size in enumerate(in_sizes)
        ]

        with tf.name_scope('params'):
            coeffs = [
                tf.Variable(tf.ones([size, 1]), name="coeff_{}".format(idx))
                for idx, size in enumerate(in_sizes)
            ]

            intercepts = [
                tf.Variable(tf.zeros([1, ]), name="intercept_{}".format(idx))
                for idx, _ in enumerate(in_sizes)
            ]

        out_layers = [
            tf.add(tf.matmul(in_layers[0], coeff), intercept, name="output_{}".format(idx))
            for idx, (coeff, intercept) in enumerate(zip(coeffs, intercepts))
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, out_size])
            for idx, out_size in enumerate(out_sizes)
        ]

        #
        # Set up loss
        #

        loss = tf.reduce_sum(
            [loss_mse(*pair) for pair in zip(out_layers, target_layers)]
        )

        return in_layers, out_layers, target_layers, loss

    def setup_training_step(self, params):
        """Set up loss and training step"""

        # Import params
        learning_rate = params.get('learning_rate', 0.1)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step
