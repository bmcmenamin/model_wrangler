"""Module contains common tensorflow operations
"""
from multiprocessing import cpu_count
import tensorflow as tf

#
# Session config functions
#

def set_session_params(cfg_params=None):
    """Set session configuration. Use this to switch between CPU and GPU tasks
    """
    if cfg_params is None:
        cfg_params = {}

    return tf.ConfigProto(**cfg_params)

def set_max_threads(sess_cfg, max_threads=None):
    """Set max threads used in session
    """
    if max_threads is None:
        max_threads = cpu_count()

    sess_cfg.intra_op_parallelism_threads = max_threads
    sess_cfg.inter_op_parallelism_threads = max_threads
    sess_cfg.allow_soft_placement = True
    return sess_cfg


#
# Loss functions
#

def layer_logits(layer, pad=1.0e-8):
    """convert probabilities to logits
    """
    if pad > 0.5 or pad < 0.0:
        raise ValueError(
            'Logit pad should be in interval (0, 0.5),',
            'but somehow you have {}'.format(pad)
            )
    pad_layer = (1 - 2*pad) * layer + pad
    logit_values = tf.log(pad_layer) - tf.log(1 - pad_layer)
    return logit_values

def loss_mse(observed, actual):
    """Mean squared error loss
    """
    return tf.reduce_sum(tf.squared_difference(observed, actual))

def loss_sigmoid_ce(observed, actual):
    """Calculate sigmoid cross entropy loss
    """
    per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=actual,
        logits=observed
    )

    per_batch_loss = tf.reduce_sum(per_sample_loss)

    return per_batch_loss

def loss_softmax_ce(observed, actual):
    """Calculate softmax cross entropy loss
    """
    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=actual,
        logits=observed
    )

    per_batch_loss = tf.reduce_sum(per_sample_loss)

    return per_batch_loss

def accuracy(observed, actual):
    """Accuracy for one-hot encoded categories
    """
    is_correct = tf.equal(tf.argmax(observed, 1), tf.argmax(actual, 1))
    accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))
    return accuracy

