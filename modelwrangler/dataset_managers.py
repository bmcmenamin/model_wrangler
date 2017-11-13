"""
This module has the tools that do dataset management
"""

# C'mon pylint, X and y is a perfectly acceptable names here...
# pylint: disable=C0103

import sys
import logging
import random
from collections import Iterable

if sys.version_info.major == 2:
    from itertools import izip_longest as zip_longest
    from itertools import izip as zip
elif sys.version_info.major == 3:
    from itertools import zip_longest
else:
    raise ValueError('wtf?!?')

import numpy as np


LOGGER = logging.getLogger(__name__)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(
    logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
)
LOGGER.addHandler(h)
LOGGER.setLevel(logging.DEBUG)


def random_chunk_generator(iterable, block_size):
    """Collect data into fixed-length chunks or blocks
    
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    https://docs.python.org/3/library/itertools.html
    """
    random.shuffle(iterable)
    args = [iter(iterable)] * block_size
    return zip_longest(*args, fillvalue=None)


def pad_list(in_list, pad_size):
    """
    Make a list a little longer by appending a randomly sampled
    items from itselt
    """
    new_list = list(in_list)
    pad_values = np.random.choice(in_list, pad_size, replace=True)
    new_list.extend(pad_values)
    return new_list


def get_groups(output_data):
    """
    Given a dataset of output variables, figure out how many
    distinct groups/categories occur, and create a dict that maps
    each category (as a tuple) to indices where it is found
    """
    group_to_idx = {}

    for idx in range(output_data.shape[0]):
        group = tuple(output_data[idx, :])

        if group in group_to_idx:
            group_to_idx[group].append(idx)
        else:
            group_to_idx[group] = [idx]
    return group_to_idx


def random_split_list(in_list, split_proportion):
    """Randomly divide list into two parts"""

    if split_proportion >= 1.0 or split_proportion < 0.0:
        raise ValueError(
            'split_proportion should be in interval (0, 1.0],',
            'but you have {}'.format(split_proportion)
            )

    cutpoint = int(len(in_list) * split_proportion)
    random.shuffle(list(in_list))

    list_0 = in_list[cutpoint:]
    list_1 = in_list[:cutpoint]
    return list_0, list_1


def flatten_lists(list_of_lists):
    """Recursively flatten a list of lists"""

    if not isinstance(list_of_lists, Iterable):
        return [list_of_lists]

    flat = []
    for item in list_of_lists:
        flat.extend(flatten_lists(item))

    return flat


class DatasetManager(object):
    """
    Load a dataset and manage how sample holdout and how samples are batch'd
    for model training

    Initialize with arrays:
     `X` is num_samples by input_dimension
     `y` is num_samples by output_dimension
     `categorical` is a bool that tells you whether the dataset has categorical
      data that we can use for stratified sampling [default: True]
     `holdout_prop` is a float that tells us what proportion of data to hold out
      for validation

    The training method will use `get_batches` to pull samples of data out of this
    manager, so you'll want to redefine that for new datasets types
    """

    def __init__(self, X, y, categorical=False, holdout_prop=0.0):

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Input array and output array have different numbers',
                'of samples: ({}, {})'.format(X.shape[0], y.shape[1])
                )

        self.X = X
        self.y = y

        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

        LOGGER.info('Input data size (%d, %d)', self.X.shape[0], self.X.shape[1])

        if not holdout_prop:
            holdout_prop = 0.0

        self.groups = {}
        self.groups_holdout = {}

        if categorical:
            self.groups = get_groups(self.y)
            LOGGER.info('Input has %d groups', len(self.groups))
        else:
            self.groups = {None: range(X.shape[0])}

        for grp in self.groups:
            idx_list = self.groups[grp]
            idx_list0, idx_list1 = random_split_list(idx_list, holdout_prop)
            self.groups[grp] = idx_list0
            self.groups_holdout[grp] = idx_list1

        self.nsamp_train = sum([len(g) for g in self.groups.values()])
        self.nsamp_holdout = sum([len(g) for g in self.groups_holdout.values()])
        LOGGER.info('Num training samples %d', self.nsamp_train)
        LOGGER.info('Num holdout samples %d', self.nsamp_holdout)

    def get_batches(self, pos_classes=None, batch_size=256, **kwargs):
        """
        This function looks at whether you've sepcifid positive classes to figure
        out if you want balanced or stratified categorical sampling
        """
        return self.random_batches(batch_size=batch_size)

    def _return_idx(self, idx):
        idx = [i for i in idx if i is not None]
        if idx:
            subset_X = np.take(self.X, idx, axis=0)
            subset_y = np.take(self.y, idx, axis=0)
            return subset_X, subset_y

    def get_holdout_samples(self):
        """Return the holdout data"""

        all_idx = flatten_lists(self.groups_holdout.values())
        return self._return_idx(all_idx)

    def random_batches(self, batch_size=256):
        """Generate random batches of size`batch_size`"""

        all_idx = flatten_lists(self.groups.values())

        for batch_idx in random_chunk_generator(all_idx, batch_size):
            yield self._return_idx(batch_idx)


class CategoricalDataManager(DatasetManager):
    """Turn categorical data into batches"""

    def __init__(self, X, y, holdout_prop=None):
        super(CategoricalDataManager, self).__init__(
            X, y,
            categorical=True,
            holdout_prop=holdout_prop
        )

    def get_batches(self, pos_classes=None, batch_size=256, **kwargs):
        """
        This function looks at whether you've sepcifid positive classes to figure
        out if you want balanced or stratified categorical sampling
        """

        if pos_classes:
            return self.balanced_batches(pos_classes=pos_classes, batch_size=batch_size)

        return self.stratified_batches(batch_size=batch_size)

    def balanced_batches(self, pos_classes=None, batch_size=256):
        """
        Generate batches where the groups listed in `pos_classes` occur with the
        same frequency as all other classes combined. Useful in the case of 
        class imbalances.
        """

        pos_samples = []
        neg_samples = []

        for g in self.groups:
            if g in pos_classes:
                pos_samples.extend(self.groups[g])
            else:
                neg_samples.extend(self.groups[g])

        size_diff = len(pos_samples) - len(neg_samples)
        if size_diff > 0:
            neg_samples = pad_list(neg_samples, size_diff)
        elif size_diff < 0:
            pos_samples = pad_list(pos_samples, -size_diff)

        sample_iterator = zip(
            random_chunk_generator(neg_samples, batch_size // 2),
            random_chunk_generator(pos_samples, batch_size // 2)
            )

        for batch_idx_pair in sample_iterator:
            batch_idx = flatten_lists(batch_idx_pair)
            yield self._return_idx(batch_idx)

    def stratified_batches(self, batch_size=256):
        """Generate batches with stratified sampling of groups"""

        num_batches = int(np.ceil(self.nsamp_train / (1.0*batch_size)))

        group_iters = {}
        for grp in self.groups:
            group_iters[grp] = random_chunk_generator(
                self.groups[grp],
                len(self.groups[grp]) // num_batches
            )

        for _ in range(num_batches):
            batch_idx = flatten_lists(map(next, group_iters.values()))
            yield self._return_idx(batch_idx)


class TimeseriesDataManager(DatasetManager):
    """Class for handling timeseries data"""

    def __init__(self, ts, holdout_prop=None):

        X, y = self.timeseries_to_trainingpairs(ts)

        # Figure out how to do holdout splitting...

        super(TimeseriesDataManager, self).__init__(
            X, y,
            categorical=False,
            holdout_prop=0.0)

        self.batches = self.random_batches
        raise NotImplementedError

    def timeseries_to_trainingpairs(self, timeseries):
        """
        divide a timeseries into X, y training pairs so something like an LSTM
        can be trained to predict timeseries
        """
