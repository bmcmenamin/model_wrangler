"""Module has a class that implements unit testing for models as described 
in this post by Chase Roberts:
  https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
"""

import numpy as np
import tensorflow as tf

from model_wrangler.dataset_managers import DatasetManager

def _get_shapes(tf_model):

    input_shape = [[i.value for i in j.get_shape()[1:]] for j in tf_model.inputs]
    output_shape = [[i.value for i in j.get_shape()[1:]] for j in tf_model.outputs]
    return input_shape, output_shape

def _make_dummy_data(input_sizes, num_samples):
    dummy_input = [np.ones([num_samples] + i) for i in input_sizes]
    return dummy_input


class ModelTester(object):
    """Run standard unit tests on a model"""

    def __init__(self, model_wrangler):

        self.mw = model_wrangler
        self.in_shape, self.out_shape = _get_shapes(model_wrangler.tf_mod)

        self.test_loss(num_samples=40)
        self.test_trainable(num_samples=40)

    def test_loss(self, num_samples=40):
        """Test that the loss is non-zero"""

        self.mw.initialize()
        dummy_input = _make_dummy_data(self.in_shape, num_samples)
        dummy_output = _make_dummy_data(self.out_shape, num_samples)
        for d in dummy_output:
            d[::2, ...] = 0.0

        feed_dict = self.mw.make_data_dict(dummy_input, dummy_output, is_training=True)
        loss = self.mw.sess.run(self.mw.tf_mod.loss, feed_dict=feed_dict)

        if loss == 0:
            raise ValueError("Model error equals zero. That ain't right")
        else:
            print("Model error not equal to 0. That's a good thing.")

    def test_trainable(self, num_samples=40):
        """Test that all params reachable via backprop"""

        self.mw.initialize()

        dummy_input = _make_dummy_data(self.in_shape, num_samples)
        dummy_output = _make_dummy_data(self.out_shape, num_samples)
        for d in dummy_output:
            d[::2, ...] = 0.0
        dummy_dm1 = DatasetManager(dummy_input, dummy_output)
        dummy_dm2 = DatasetManager(dummy_input, dummy_output)
        self.mw.add_data(dummy_dm1, dummy_dm2)

        before_vals = [
            self.mw.sess.run(var)
            for var in self.mw.tf_mod.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ]

        self.mw.train()

        after_vals = [
            self.mw.sess.run(var)
            for var in self.mw.tf_mod.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ]

        something_changed = False
        for before, after in zip(before_vals, after_vals):
            if (before != after).any():
                something_changed = True
                break

        if not something_changed:
            raise ValueError("No traininble variables changed after training. That ain't right.")
        else:
            print("Things in the model changed after training. That's a good thing.")


