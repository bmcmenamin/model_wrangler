"""Module has a class that implements unit testing for models as described 
in this post by Chase Roberts:
  https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
"""

import numpy as np
import tensorflow as tf

from .tf_ops import make_data_dict

def _get_shapes(tf_model):

    if isinstance(tf_model.input, list):
        input_shape = [[i.value for i in j.get_shape()[1:]] for j in tf_model.input]
    else:
        input_shape = [i.value for i in tf_model.input.get_shape()[1:]]

    output_shape = [i.value for i in tf_model.output.get_shape()[1:]]
    return input_shape, output_shape

def _make_dummy_input(input_size, num_samples):

    if isinstance(input_size[0], list):
        dummy_input = [np.ones([num_samples] + i) for i in input_size]
    else:
        dummy_input = np.ones([num_samples] + input_size)
    return dummy_input

class ModelTester(object):
    """Run standard unit tests on a model"""

    def __init__(self, mw_model_class):
        self.model_class = mw_model_class
        self.test_loss(num_samples=10)
        self.test_trainable(num_samples=10)

    def test_loss(self, num_samples=10):
        """Test that the loss is non-zero"""

        model = self.model_class()

        in_shape, out_shape = _get_shapes(model.tf_mod)

        dummy_input = _make_dummy_input(in_shape, num_samples)
        dummy_output = 0.5 * np.ones([num_samples] + out_shape)

        feed_dict = make_data_dict(
            model.tf_mod,
            dummy_input,
            dummy_output,
            is_training=True
        )

        loss = model.sess.run(model.tf_mod.loss, feed_dict=feed_dict)

        if loss == 0:
            raise ValueError("Model error equals zero. That ain't right")
        else:
            print("Model error not equal to 0. That's a good thing.")


    def test_trainable(self, num_samples=10):
        """Test that all params reachable via backprop"""

        model = self.model_class()

        in_shape, out_shape = _get_shapes(model.tf_mod)
        dummy_input = _make_dummy_input(in_shape, num_samples)
        dummy_output = 0.5 * np.ones([num_samples] + out_shape)

        model.initialize()
        before_vals = [
            model.sess.run(var)
            for var in model.tf_mod.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ]

        model.train(dummy_input, dummy_output)

        after_vals = [
            model.sess.run(var)
            for var in model.tf_mod.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ]

        something_changed = False
        for b, a in zip(before_vals, after_vals):
            if (b != a).any():
                something_changed = True
                break

        if not something_changed:
            raise ValueError("No traininble variables changed after training. That ain't right.")
        else:
            print("Things in the model changed after training. That's a good thing.")


