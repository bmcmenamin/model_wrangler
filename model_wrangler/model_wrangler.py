"""Module implements the ModelWrangler object"""

import sys
import os
import logging
import json
import pickle

from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

LOGGER = logging.getLogger(__name__)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(
    logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
)
LOGGER.addHandler(h)
LOGGER.setLevel(logging.DEBUG)


def set_session_params(cfg_params=None):
    """Set session configuration. Use this to switch between CPU and GPU tasks"""

    if cfg_params is None:
        cfg_params = {}

    return tf.ConfigProto(**cfg_params)


def set_max_threads(sess_cfg, max_threads=None):
    """Set max threads used in session"""

    if max_threads is None:
        max_threads = cpu_count()

    sess_cfg.intra_op_parallelism_threads = max_threads
    sess_cfg.inter_op_parallelism_threads = max_threads
    sess_cfg.allow_soft_placement = True
    return sess_cfg



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

    def make_data_dict(self, x_data, y_data, is_training=False):
        """Make a dict of data for feed_dict"""

        data_dict = {}
        if y_data is not None:
            for tf_layer, _y in zip(self.tf_mod.targets, y_data):
                data_dict[tf_layer] = _y

        if x_data is not None:
            for tf_layer, _x in zip(self.tf_mod.inputs, x_data):
                data_dict[tf_layer] = _x

        if is_training is not None:
            data_dict[self.tf_mod.is_training] = is_training

        return data_dict

    def __init__(self, model_class, model_params):
        """Initialize a tensorflow model"""

        self.training_data = None
        self.holdout_data = None
        self.training_gen = None
        self.holdout_gen = None
        self.training_params = {}

        self.model_params = model_params
        self.model_params['model_class'] = model_class

        self.tf_mod = model_class(model_params)

        self.session_params = set_max_threads(set_session_params({
	    'log_device_placement': True
	}))
        self.sess = self.new_session()
        
        self.initialize()

    def add_data(self, training_dataset, holdout_dataset):
        """Add datasets for training/testing"""
        self.training_data = training_dataset
        self.holdout_data = holdout_dataset
        return self

    def add_train_params(self, training_params):
        """Add datasets for training/testing"""
        self.training_params = training_params
        return self

    def _restore_train_params(self):
        # Restore training parameters (if they exist) training params

        try:
            with open(''.join([self.model_params['path'], 'training_params.pickle']), 'rb') as file:
                train_params = pickle.load(file)

            self.add_train_params(train_params)
        except FileNotFoundError:
            pass

    def save(self, iteration):
        """Save model parameters in a JSON and model weights in TF format"""

        LOGGER.info('Saving weights file in %s', self.model_params['path'])

        try:
            os.mkdir(self.model_params['path'])
        except FileExistsError:
            pass

        # Save model weights
        self.model_params['meta_filename'] = self.tf_mod.saver.save(
            self.sess,
            save_path=os.path.join(
                self.model_params['path'],
                self.model_params['name']
            ),
            global_step=iteration
        )

        # Save model parameters, training parameters
        with open(os.path.join(self.model_params['path'], 'model_params.pickle'), 'wb') as file:
            pickle.dump(self.model_params, file)

        if self.training_params:
            with open(os.path.join(self.model_params['path'], 'train_params.pickle'), 'wb') as file:
                pickle.dump(self.training_params, file)

    @classmethod
    def load(cls, param_file):
        """restore a saved model given the path to a paramter JSON file"""

        # load model params
        with open(param_file, 'rb') as file:
            model_params = pickle.load(file)

        # initialize a new model, restore its weights
        new_model = cls(model_params['model_class'], model_params)

        #last_checkpoint = tf.train.latest_checkpoint(new_model.model_params[path])
        last_checkpoint = model_params['meta_filename'] 
        new_model.tf_mod.saver = tf.train.import_meta_graph(last_checkpoint + '.meta')
        new_model.tf_mod.saver.restore(new_model.sess, last_checkpoint)

        new_model._restore_train_params()

        return new_model


    def predict(self, input_x):
        """Get activations for every layer given an input matrix, input_x"""

        data_dict = self.make_data_dict(input_x, None, is_training=False)
        vals = self.sess.run(self.tf_mod.outputs, feed_dict=data_dict)
        return vals

    def embed(self, input_x):
        """Get embeddings for an input value, input_x"""

        if self.tf_mod.embeds is None:
            raise AttributeError('the embedding layers are not defined in the model')

        data_dict = self.make_data_dict(
            input_x,
            None, is_training=False)

        vals = self.sess.run(
            self.tf_mod.embeds,
            feed_dict=data_dict
        )

        return vals

    def score(self, input_x, target_y, score_func=None):
        """Measure model's current performance
        for a set of input_x and target_y using some scoring function
        `score_func` (defults to model loss function)
        """

        data_dict = self.make_data_dict(input_x, target_y, is_training=False)

        if score_func is None:
            score_func = self.tf_mod.loss
        else:
            with self.tf_mod.graph.as_default():
                score_func = tf.reduce_mean([score_func(*pair) for pair in zip(self.tf_mod.outputs, self.tf_mod.targets)])
        val = score_func.eval(feed_dict=data_dict, session=self.sess)

        return val

    def feature_importance(self, input_x, target_y, score_func=None):
        """Calculate feature importances"""

        # which layer has the features you care about?
        feature_layer_idx = 0

        data_dict = self.make_data_dict(input_x, target_y, is_training=False)

        if score_func is None:
            score_func = self.tf_mod.loss

        grad_wrt_input = tf.gradients(
            score_func,
            self.tf_mod.inputs
        )

        grad_wrt_input_vals = self.sess.run(
            grad_wrt_inputs,
            feed_dict=data_dict
        )[feature_layer_idx]

        importance = np.mean(grad_wrt_input_vals**2, axis=0, keepdims=True)

        return importance

    def _run_epoch(self):
        """Run an epoch of training"""

        train_verbose = self.training_params.get('verbose', True)
        train_verbose_interval = self.training_params.get('interval', 100)

        for batch_counter, (train_in, train_out) in enumerate(self.training_gen):

            if batch_counter >= self.training_params.get('epoch_length', np.inf):
                break

            data_dict = self.make_data_dict(train_in, train_out, is_training=True)
            self.sess.run(self.tf_mod.train_step, feed_dict=data_dict)

            if train_verbose and ((batch_counter % train_verbose_interval) == 0):
                ho_in, ho_out = next(self.holdout_gen)

                data_dict = self.make_data_dict(ho_in, ho_out, is_training=False)

                # Write training stats to tensorboard
                self.tf_mod.tb_writer.add_summary(
                    self.sess.run(self.tf_mod.tb_stats, feed_dict=data_dict),
                    batch_counter
                )

                # logging elsewhere
                train_error = self.score(train_in, train_out)
                holdout_error = self.score(ho_in, ho_out)
                LOGGER.info("Batch %d: Training score = %0.6f", batch_counter, train_error)
                LOGGER.info("Batch %d: Holdout score = %0.6f", batch_counter, holdout_error)


    def train(self):
        """
        Run a a bunch of training batches
        on the model using a bunch of input_x, target_y
        """
        num_epochs = self.training_params.get('num_epochs', 1)
        epoch_length = self.training_params.get('epoch_length', None)
        batch_size = self.training_params.get('batch_size', 32)

        self.training_gen = self.training_data.get_next_batch(
            batch_size=batch_size, eternal=epoch_length is not None)

        self.holdout_gen = self.holdout_data.get_next_batch(
            batch_size=batch_size, eternal=True)

        try:
            for epoch in range(num_epochs):
                LOGGER.info('Starting Epoch %d', epoch)
                self._run_epoch()
                self.save(epoch)

                if not epoch_length:
                    self.training_gen = self.training_data.get_next_batch(
                        batch_size=batch_size, eternal=False)

                    self.holdout_gen = self.holdout_data.get_next_batch(
                        batch_size=batch_size, eternal=True)


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

