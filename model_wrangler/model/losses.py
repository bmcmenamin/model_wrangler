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


def tensor_diff(tensor_0, tensor_1):
    return tf.reduce_sum(tf.square(tensor_0 - tensor_1), 1)

def siamese_embedding_loss(embed_0, embed_1, is_match, margin=1.0):
    """Loss for two batches of siamese embeddings
    Args:
        embed_0, embed_1, batches of embeddings
        is_match tensor of 0/1 indicating when pairs are matched
    """

    diff = tensor_diff(embed_0, embed_1)
    diff_sqrt = tf.sqrt(diff)

    loss_per_sample = is_match * tf.square(tf.maximum(0., margin - diff_sqrt)) + (1 - is_match) * diff
    loss = 0.5 * tf.reduce_mean(loss_per_sample)

    return loss

def triplet_embedding_loss(embed_0, embed_1, embed_anchor, margin=1.0):
    """Loss for two siamese embeddings"""

    diff_0 = tensor_diff(embed_0, embed_anchor)
    diff_1 = tensor_diff(embed_1, embed_anchor)

    loss_per_batch = tf.maximum(0., margin + diff_0 - diff_1)
    loss = tf.reduce_mean(loss_per_batch)

    return loss
