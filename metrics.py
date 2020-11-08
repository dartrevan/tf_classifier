import tensorflow as tf


def metrics(label_ids, predictions):
    return {'Accuracy': tf.metrics.accuracy(label_ids, predictions)}
