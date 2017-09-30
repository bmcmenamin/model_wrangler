"""
This module has the tools that do dataset management
"""

import sys
import logging
import random

if sys.version_info.major == 2:
    from itertools import izip_longest as zip_longest
    from itertools import izip as zip
elif sys.version_info.major == 3:
    from itertools import zip_longest
else:
    raise ValueError('wtf?!?')

import numpy as np


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
    new_list = in_list.copy()
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

    for idx, group in enumerate(map(tuple, output_data)):
        if group in group_to_idx:
            group_to_idx[group].append(idx)
        else:
            group_to_idx[group] = [idx]

    return group_to_idx

def random_split_list(in_list, split_proportion):
    """Randomly divide list into two parts
    """
    if split_proportion >= 1.0 or split_proportion < 0.0:
        raise ValueError(
            'split_proportion should be in interval (0, 1.0],',
            'but you have {}'.format(split_proportion)
            )

    cutpoint = len(in_list) * split_proportion
    random.shuffle(in_list)

    list_0 = in_list[cutpoint:]
    list_1 = in_list[:cutpoint]
    return list_0, list_1

def flatten_lists(list_of_lists):
    """Recursively flatten a list of lists
    """
    flat = []
    for item in list_of_lists:

        if isinstance(item, tuple):
            item = list(item)

        if isinstance(item, list):
            item = flatten_list(item)
            flat.extend(item)
        else:
            flat.append(item)

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
    """

    # C'mon pylint, X and y is a perfectly acceptable names here...
    # pylint: disable=C0103

    def __init__(self, X, y, categorical=True, holdout_prop=None):

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Input array and output array have different numbers',
                'of samples: ({}, {})'.format(X.shape[0], y.shape[1])
                )

        self.X = X
        self.y = y
        logging.info('Input data size (%d, %d)', self.X.shape[0], self.X.shape[1])

        if not holdout_prop:
            holdout_prop = 0.0

        self.groups = {}
        self.groups_holdout = {}

        if categorical:
            self.groups = get_groups(y)
            logging.info('Input has % groups', len(self.groups))
        else:
            self.groups = {None: range(X.shape[0])}

        for grp in self.groups:
            idx_list = self.groups[grp]
            idx_list0, idx_list1 = random_split_list(idx_list, holdout_prop)
            self.groups[grp] = idx_list0
            self.groups_holdout[grp] = idx_list1

        self.nsamp_train = sum([len(g) for g in self.groups.values()])
        self.nsamp_holdout = sum([len(g) for g in self.groups_holdout.values()])
        logging.info('Num training samples %d', len(self.nsamp_train))
        logging.info('Num holdout samples %d', len(self.nsamp_holdout))


    def get_holdout_samples(self):
        """Return the holdout data
        """
        all_idx = flatten_lists(self.groups_holdout.values)
        return self.X[all_idx, :], self.y[all_idx]

    def random_batches(self, batch_size=256):
        """Generate random batches of size`batch_size`
        """
        all_idx = flatten_lists(self.groups.values)

        for batch_idx in random_chunk_generator(all_idx, batch_size):
            yield self.X[batch_idx, :], self.y[batch_idx]

    def stratified_batches(self, batch_size=256):
        """Generate batches with stratified sampling of groups
        """

        num_batches = self.nsamp_train // batch_size

        group_iters = {}
        for grp in self.groups:
            group_iters[grp] = random_chunk_generator(
                self.groups[grp],
                len(self.groups[grp]) // num_batches
                )

        for _ in range(num_batches):
            batch_idx = flatten_lists(map(next, group_iters.values()))
            yield self.X[batch_idx, :], self.y[batch_idx]

    def balanced_batches(self, pos_classes, batch_size=256):
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
            yield self.X[batch_idx, :], self.y[batch_idx]
