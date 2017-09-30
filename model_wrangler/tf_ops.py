"""Module contains common tensorflow operations
"""
import tensorflow as tf

def layer_logits(layer, pad=1.0e-8):
    """convert probabilities to logits"""

    if pad > 0.5 or pad < 0.0:
        raise ValueError(
            'Logit pad should be in interval (0, 0.5),',
            'but somehow you have {}'.format(pad)
            )

    pad_layer = (1 - 2*pad) * layer + pad
    return tf.log(pad_layer)



#
# Loss functions
#

def _loss(function, observed, actual):
    """Calculate loss with arbitrary loss func
    """

    per_sample_loss = function(actual, observed)
    per_batch_loss = tf.reduce_mean(per_sample_loss)
    return per_batch_loss

def loss_mse(observed, actual):
    """Mean squared error loss
    """
    return _loss(tf.metric.mean_squared_error, observed, actual)

def loss_sigmoid_ce(observed, actual):
    """Calculate sigmoid cross entropy loss
    """

    per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=actual,
        logits=layer_logits(observed)
    )

    per_batch_loss = tf.reduce_mean(per_sample_loss)

    return per_batch_loss

def loss_softmax_ce(observed, actual):
    """Calculate softmax cross entropy loss
    """

    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=actual,
        logits=layer_logits(observed)
    )

    per_batch_loss = tf.reduce_mean(per_sample_loss)

    return per_batch_loss


