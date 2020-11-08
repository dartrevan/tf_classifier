from utils import _preprocess, _preprocess_bunch
import tensorflow as tf
import numpy as np


def sequence_initializer(path, params):
    def tf_preprocess(text):
        return tf.py_func(
            lambda txt: _preprocess(txt,  max_len=params['max_length'], wordlevel=params['wordlevel'], check_spelling=params['spell_checking']),
            [text],
            tf.string
        )

    def _set_shape(t, mlen):
        t.set_shape([mlen])
        return t

    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(tf_preprocess)
    return dataset.map(lambda t: _set_shape(t, params['max_length']))


def bunch_of_sequences_initializer(path, params):
    def tf_preprocess(text):
        return tf.py_func(
            lambda txt: _preprocess_bunch(
                txt, max_len=params['max_length'], wordlevel=params['wordlevel'], 
                check_spelling=params['spell_checking'], bunch_size=params['bunch_size']
            ),
            [text],
            tf.string,
            name='bunch_of_seq'
        )

    def _set_shape(t):
        t.set_shape([params['bunch_size'], params['max_length']])
        return t

    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(tf_preprocess)
    return dataset.map(_set_shape)


def numeric_initializer(path, params):
    dataset = np.loadtxt(path).astype(params['dtype'])
    return tf.data.Dataset.from_tensor_slices(dataset)


def numeric_sequence_initializer(path, params):
    def tf_preprocess(text):
        return tf.py_func(
            lambda txt: _preprocess_bunch(
                txt, max_len=1, wordlevel=True,
                check_spelling=False, padding_value='0', bunch_size=params['bunch_size']),
            [text],
            tf.string,
            name='numeric_seq'
        )

    def _set_shape(t):
        t.set_shape([params['bunch_size']])
        return t

    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(tf_preprocess)
    dataset = dataset.map(lambda x: tf.string_to_number(x, out_type=tf.int32))
    return dataset.map(_set_shape)


def textline_initializer(path, params):
    return tf.data.TextLineDataset(path)


ds_initializers = {
    'sequence': sequence_initializer,
    'textline': textline_initializer,
    'numeric': numeric_initializer,
    'bunch_of_sequences': bunch_of_sequences_initializer,
    'numeric_sequence': numeric_sequence_initializer
}


def get_input_fn(features, target=None, batch_size=128, train=True):

    def input_fn():
        # initializing target
        target_ds = None
        if target is not None:
            target_ds = tf.data.TextLineDataset(target['path']) #sequence_initializer(target['path'], target['params'])

        # initializing features
        features_ds = {}
        for feature_name, feature_meta in features.items():
            feature_type = feature_meta['type']
            feature_path = feature_meta['path']
            feature_params = feature_meta['params']
            ds_initializer = ds_initializers[feature_type]
            features_ds[feature_name] = ds_initializer(feature_path, feature_params)

        if target is not None:
            dataset = tf.data.Dataset.zip((features_ds, target_ds))
        else:
            dataset = tf.data.Dataset.zip(features_ds)

        if train:
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(batch_size*3)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn
