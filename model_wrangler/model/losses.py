import tensorflow as tf

def layer_logits(layer, pad=1.0e-8):
    """convert probabilities to logits"""

    if pad > 0.5 or pad < 0.0:
        raise ValueError(
            'Logit pad should be in interval (0, 0.5),',
            'but somehow you have {}'.format(pad)
            )
    pad_layer = (1 - 2 * pad) * layer + pad
    logit_values = tf.log(pad_layer) - tf.log(1 - pad_layer)
    return logit_values


def loss_mse(observed, actual):
    """Mean squared error loss"""
    numel = tf.reduce_prod(tf.size(observed))
    return tf.reduce_sum(tf.squared_difference(observed, actual)) / tf.cast(numel, tf.float32)


def loss_sigmoid_ce(observed, actual):
    """Calculate sigmoid cross entropy loss"""

    per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=actual,
        logits=observed
    )
    numel = tf.reduce_prod(tf.size(observed))
    per_batch_loss = tf.reduce_sum(per_sample_loss) / tf.cast(numel, tf.float32)
    return per_batch_loss


def loss_softmax_ce(observed, actual):
    """Calculate softmax cross entropy loss"""

    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=observed,
        labels=actual
    )
    numel = tf.reduce_prod(tf.size(observed))
    per_batch_loss = tf.reduce_sum(per_sample_loss) / tf.cast(numel, tf.float32)
    return per_batch_loss


def accuracy(observed, actual):
    """Accuracy for one-hot encoded categories"""

    is_correct = tf.equal(tf.argmax(observed, axis=1), tf.argmax(actual, axis=1))
    acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return acc
