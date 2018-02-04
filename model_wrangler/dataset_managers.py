"""This module has the tools that do dataset management"""

# C'mon pylint, X and Y is a perfectly acceptable names here...
# pylint: disable=C0103

import sys
import logging

import random

from itertools import islice, cycle
from collections import Iterable

from abc import ABC, abstractmethod

import numpy as np

LOGGER = logging.getLogger(__name__)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(
    logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
)
LOGGER.addHandler(h)
LOGGER.setLevel(logging.DEBUG)


class BaseDatasetManager(ABC):
    """
    Abstract class used to read in datasets and serve up data samples
    for batched training and validation.

    The training method will use `get_next_batch` and `get_next_holdout_batch` 
    to pull samples of data out of this manager, so you'll want to
    redefine those for new datasets types
    """

    @staticmethod
    def _force_to_generators(x_in):
        """Force an input to be a generator"""

        if isinstance(x_in, np.ndarray):
            if len(x_in.shape) == 1:
                x_in = x_in.reshape(-1, 1)
            return (row for row in x_in)

        elif isinstance(x_in, Iterable):
            return (row for row in x_in)

        else:
            raise AttributeError('Input is not an array or an Iterable')

    def _shuffle_data(self, X, Y):
        """Suffle input/output sample order"""

        new_order = list(range(len(X[0])))
        random.shuffle(new_order)

        X = [[x[i] for i in new_order] for x in X]
        Y = [[y[i] for i in new_order] for y in Y]

        return X, Y

    def _yield_batches(self, X, Y, batch_size):
        for idx in range(0, len(X[0]), batch_size):
            X_batch = [x[idx:(idx + batch_size)] for x in X]
            Y_batch = [y[idx:(idx + batch_size)] for y in Y]
            yield X_batch, Y_batch

    def _combine_list_of_generators(self, gen_list):
        """Turn a list of generators into a single generator
        that returns a list of samples"""

        out_list = [
            list(islice(gen, self.cache_size))
            for gen in gen_list
        ]

        yield out_list

    def _input_to_generators(self, input, eternal=False):
        gen = self._combine_list_of_generators(
            [self._force_to_generators(x) for x in input]
        )

        if eternal:
            gen = cycle(gen)

        return gen

    def __init__(self, X, Y, cache_size=2056):
        """
        Args:
          X: is a list of (num_samples x input_dimension) arrays and/or iterables
          Y: is a list of (num_samples x output_dimension) arrays and/or iterables

          cache_size: is the number of samples to cache internally
            from the input generators. This list should be larger than the
            batch sizes used in training because it's what gets randomly
            shuffled across epochs
        """

        self.num_inputs = len(X)
        self.num_outputs = len(Y)
        self.cache_size = 2056

        self.X = X
        self.Y = Y

        LOGGER.info('Dataset has %d inputs', self.num_inputs)
        LOGGER.info('Dataset has %d outputs', self.num_outputs)

    @abstractmethod
    def get_next_batch(self, batch_size=32):
        """
        This generator should yield batches of training data

        Args:
            batch_size: int for number of samples in batch
        Yields:
            X, Y: lists of input/output samples
        """
        yield [1], [1]



class DatasetManager(BaseDatasetManager):
    """Just a plain ol' vanilla Dataset Manager that returns
    batches of data in nearly-random order"""


    def get_next_batch(self, batch_size=32, eternal=False):
        """
        This generator should yield batches of training data

        Args:
            batch_size: int for number of samples in batch
        Yields:
            X, Y: lists of input/output samples
        """

        X_gen = self._input_to_generators(self.X, eternal=eternal)
        Y_gen = self._input_to_generators(self.Y, eternal=eternal)

        for X, Y in zip(X_gen, Y_gen):
            X, Y = self._shuffle_data(X, Y)
            for x, y in self._yield_batches(X, Y, batch_size):
                yield x, y



class SequentialDatasetManager(BaseDatasetManager):
    """Dataset Manager for handling sequential inputs like
    timeseries or text sequences in an RNN"""


    def get_next_batch(self, batch_size=32, eternal=False):
        """
        This generator should yield batches of training data

        Args:
            batch_size: int for number of samples in batch
            eternal: this generator runs forever and ever
        Yields:
            X, Y: lists of input/output samples
        """

        X_gen = self._input_to_generators(self.X, eternal=eternal)
        Y_gen = self._input_to_generators(self.Y, eternal=eternal)

        for X, Y in zip(X_gen, Y_gen):
            for x, y in self._yield_batches(X, Y, batch_size):
                yield x, y


class BalancedDatasetManager(BaseDatasetManager):
    """Balance the datasets so there are equal numbers of
    positive and negative classes for training

    positive classes are defined using by `positive_classes`, which
    is a list with an item corresponding to each output in Y. The ith item
    in the list will be used to determine whether the sample in the ith
    output is in the positive class. A sample is labelled as 'positive' is
    *any* of its outputs are positive-class.

    The list can contain:
        - a function that returns a bool
        - a list or set of 'allowed' values
        - a None indicating that the output is not used to 
            determine pos/neg class
    """

    def _setup_positive_class_def(self, positive_classes):
        """Make sure the positive class definition is correct"""

        if len(positive_classes) != self.num_outputs:
            raise ValueError(
                'Positive classes defined for {} outputs '
                'but the data has {} outputs'.format(
                    len(positive_classes),
                    self.num_outputs
                )
            )

        for idx, func in enumerate(positive_classes):

            if func is None:
                positive_classes[idx] = lambda x: False

            elif isinstance(func, set) is None:
                positive_classes[idx] = lambda x: x in func

            elif isinstance(func, list) is None:
                positive_classes[idx] = lambda x: x in set(func)

            elif not hasattr(func, '__call__'):
                raise AttributeError(
                    'Positive class definition {}, ({}) is not valid. '
                    'Must be either None, a callable function or a list/set of '
                    'allowable values'
                    .format(idx, func)
                )

        return positive_classes

    def _find_positive_class_samples(self, positive_classes, data_in):
        """Return a list of booleans indicating whether a particular
        sample is in the positive class"""

        pos_idx = [
            any(sample) for sample in
            zip(*[map(func, data) for func, data in zip(positive_classes, data_in)])
        ]

        return pos_idx

    @staticmethod
    def pad_list(in_list, pad_size):
        """
        Make a list a longer by appending randomly sampled
        items from itself
        """
        new_list = list(in_list)
        pad_values = np.random.choice(in_list, pad_size, replace=True)
        new_list.extend(pad_values)
        return new_list

    def _shuffle_data(self, X, Y, positive_classes):
        """Suffle input/output sample order after doing up/downsampling
        to make sure that the positive/negative classes are equally
        balanced
        """

        is_pos_bool = self._find_positive_class_samples(positive_classes, Y)

        pos_idx = []
        neg_idx = []
        for idx, is_pos in enumerate(is_pos_bool):
            if is_pos:
                pos_idx.append(idx)
            else:
                neg_idx.append(idx)
        num_pos, num_neg = len(pos_idx), len(neg_idx)


        if num_pos == 0 or num_neg == 0:
            X = [[] for x in X]
            Y = [[] for y in Y]

        else:
            if num_pos > num_neg:
                neg_idx = self.pad_list(neg_idx, num_pos - num_neg)
            else:
                pos_idx = self.pad_list(pos_idx, num_neg - num_pos)

            resample_idx = neg_idx + pos_idx
            random.shuffle(resample_idx)

            X = [[x[i] for i in resample_idx] for x in X]
            Y = [[y[i] for i in resample_idx] for y in Y]

        return X, Y

    def get_next_batch(self, positive_classes, batch_size=32, eternal=False):
        """
        This generator should yield batches of training data

        Args:
            positive_classes: list of functions defining the 'positive class'
                based on each output type
            batch_size: int for number of samples in batch
            eternal: this generator runs forever and ever
        Yields:
            X, Y: lists of input/output samples
        """

        positive_classes = self._setup_positive_class_def(positive_classes)
        X_gen = self._input_to_generators(self.X, eternal=eternal)
        Y_gen = self._input_to_generators(self.Y, eternal=eternal)

        for X, Y in zip(X_gen, Y_gen):
            X, Y = self._shuffle_data(X, Y, positive_classes)
            for x, y in self._yield_batches(X, Y, batch_size):
                yield x, y
