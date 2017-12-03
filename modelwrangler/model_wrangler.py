"""Module implements the ModelWrangler object
"""
import sys
import os
import logging
import json

import numpy as np
import tensorflow as tf

import modelwrangler.tf_ops as tops
from modelwrangler.tf_models import BaseNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

LOGGER = logging.getLogger(__name__)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(
    logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
)
LOGGER.addHandler(h)
LOGGER.setLevel(logging.DEBUG)


class ModelWrangler(object):
    """
    Loads a model class that you've defined and wraps it with a bunch of helpful methods:
        `save`: save tf model to disk
        `restore`: bring a trained model back from the dead (i.e. load from disk)

        `predict`: get model activations for a given input        
        `score`: get model model loss for a set of inputs and target outputs
        `feature_importance`: estimate feature importance by looking at error
            gradients for a set of inputs and target outputs
    """

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError:
            pass

    def new_session(self):
        """Make Tensorflow session"""

        sess = tf.Session(
            graph=self.tf_mod.graph,
            config=self.session_params
        )
        return sess

    def initialize(self):
        """Initialize model weights"""

        initializer = tf.variables_initializer(
            self.tf_mod.graph.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES
                )
            )
        self.sess.run(initializer)

    def __init__(self, model_class=BaseNetwork, **kwargs):
        """Initialize a tensorflow model"""

        self.session_params = tops.set_max_threads(tops.set_session_params())
        self.params = model_class.PARAM_CLASS(**kwargs)
        self.tf_mod = model_class(self.params)
        self.sess = self.new_session()
        self.initialize()

    def save(self, iteration):
        """Save model parameters in a JSON and model weights in TF format"""

        path_parts = [
            os.path.join(self.params.path, self.params.name),
            iteration
        ]

        LOGGER.info('Saving weights file in %s', self.params.path)
        self.tf_mod.saver.save(
            self.sess,
            save_path=path_parts[0],
            global_step=path_parts[1],
        )

        self.params.meta_filename = '{}-{}'.format(*path_parts)
        self.params.save()


    @classmethod
    def load(cls, param_file):
        """restore a saved model given the path to a paramter JSON file"""
        # load model params
        with open(param_file, 'rt') as pfile:
            params = json.load(pfile)

        # initialize a new model, restore its weights
        new_model = cls(**params)

        last_checkpoint = tf.train.latest_checkpoint(new_model.params.path)
        new_model.tf_mod.saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
        new_model.tf_mod.saver.restore(new_model.sess, last_checkpoint)

        return new_model

    def predict(self, input_x):
        """Get activations for every layer given an input matrix, input_x"""

        data_dict = tops.make_data_dict(
            self.tf_mod,
            input_x,
            None,
            is_training=False
        )

        vals = self.sess.run(self.tf_mod.output, feed_dict=data_dict)

        return vals

    def score(self, input_x, target_y, score_func=None):
        """Measure model's current performance
        for a set of input_x and target_y using some scoring function
        `score_func` (defults to model loss function)
        """

        data_dict = tops.make_data_dict(
            self.tf_mod,
            input_x,
            target_y,
            is_training=False
        )

        if score_func is None:
            score_func = self.tf_mod.loss
        else:
            with self.tf_mod.graph.as_default():
                score_func = score_func(self.tf_mod.output, self.tf_mod.target)

        val = score_func.eval(feed_dict=data_dict, session=self.sess)

        return val

    def feature_importance(self, input_x, target_y, score_func=None):
        """Calculate feature importances"""

        # which layer has the features you care about?
        feature_layer_idx = 0

        data_dict = tops.make_data_dict(
            self.tf_mod,
            input_x,
            target_y,
            is_training=False
        )

        if score_func is None:
            score_func = self.tf_mod.loss

        grad_wrt_input = tf.gradients(
            score_func,
            self.tf_mod.input
        )

        grad_wrt_input_vals = self.sess.run(
            grad_wrt_input,
            feed_dict=data_dict
        )[feature_layer_idx]

        importance = np.mean(grad_wrt_input_vals**2, axis=0, keepdims=True)

        return importance


    def _run_epoch(self, sess, dataset, pos_classes):
        """Run an epoch of training"""

        batch_iterator = dataset.get_batches(
            pos_classes=pos_classes,
            batch_size=self.params.batch_size
        )

        X_holdout, y_holdout = dataset.get_holdout_samples()

        batch_counter = 0
        for X_batch, y_batch in batch_iterator:

            data_dict = tops.make_data_dict(
                self.tf_mod,
                X_batch,
                y_batch,
                is_training=True
            )

            sess.run(
                self.tf_mod.train_step,
                feed_dict=data_dict
            )

            if (batch_counter % 100) == 0:

                # Write training stats to tensorboard
                self.tf_mod.tb_writer.add_summary(
                    sess.run(self.tf_mod.tb_stats, feed_dict=data_dict),
                    batch_counter
                )

                # logging elsewhere
                train_error = self.score(X_batch, y_batch)
                holdout_error = self.score(X_holdout, y_holdout)
                LOGGER.info("Batch %d: Training score = %0.6f", batch_counter, train_error)
                LOGGER.info("Batch %d: Holdout score = %0.6f", batch_counter, holdout_error)

            batch_counter += 1

    def train(self, input_x, target_y, pos_classes=None):
        """
        Run a a bunch of training batches
        on the model using a bunch of input_x, target_y
        """

        dataset = self.tf_mod.DATA_CLASS(
            input_x, target_y,
            holdout_prop=self.params.holdout_prop)

        try:
            for epoch in range(self.params.num_epochs):
                LOGGER.info('Starting Epoch %d', epoch)
                self._run_epoch(self.sess, dataset, pos_classes)
                self.save(epoch)

        except KeyboardInterrupt:
            print('Force exiting training.')

    def get_from_model(self, name_to_find):
        """Return a piece of the model by it's name"""

        if name_to_find not in self.tf_mod.graph._names_in_use:
            raise KeyError('`{}` not in this model'.format(name_to_find))

        ops_list = [i.name for i in self.tf_mod.graph.get_operations()]
        if name_to_find in ops_list:
            tensor_item = self.tf_mod.graph.get_tensor_by_name('{}:0'.format(name_to_find))
        else:
            tensor_item = self.tf_mod.graph.get_tensor_by_name(name_to_find)

        value_item = self.sess.run(tensor_item)
        return value_item

