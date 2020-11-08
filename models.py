from utils import init_embeddings
from metrics import metrics
import tensorflow as tf
import tensorflow_hub as hub


def get_elmo():
  if elmo is None:
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
  return elmo


def lookup(x, path_to_vocablary):
    lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=path_to_vocablary,
        vocab_size=None,
        default_value=0
    )

    return lookup_table.lookup(x)


def rev_lookup(x, path_to_vocabulary):
    x = tf.cast(x, tf.int64)
    lookup_table = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=path_to_vocabulary
    )
    return lookup_table.lookup(x)


def embed(x, token_vocab_file, pretrained_embeddings_file):
    token_ids = lookup(x, token_vocab_file)
    embeddings = init_embeddings(
        path_to_vocab=token_vocab_file,
        pretrained_embeddings=pretrained_embeddings_file,
        binary=True
    )
    return tf.contrib.layers.embed_sequence(
        token_ids,
        vocab_size=embeddings.shape[0],
        embed_dim=embeddings.shape[1],
        initializer=tf.constant_initializer(embeddings),
        trainable=True
    )


def elmo_embed(tokens, lengths):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    return elmo(
            inputs={
                'tokens': tokens,
                'sequence_len': tf.clip_by_value(lengths, 0, 30)
            },
            signature="tokens",
            as_dict=True)['elmo']


def rnn_layer(inputs, lengths, mode, params, return_sequences=True, keep_prob=0.8, input_size=None):
    rnn_cell_fw = tf.nn.rnn_cell.GRUCell(num_units=params.rnn_num_units, name='gru_fw', reuse=tf.AUTO_REUSE)
    rnn_cell_bw = tf.nn.rnn_cell.GRUCell(num_units=params.rnn_num_units, name='gru_bw', reuse=tf.AUTO_REUSE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_fw,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob,
            state_keep_prob=keep_prob,
            variational_recurrent=input_size is not None,
            input_size=input_size,
            dtype=tf.float32
        )
        rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_bw,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob,
            state_keep_prob=keep_prob,
            variational_recurrent=input_size is not None,
            input_size=input_size,
            dtype=tf.float32
        )

    rnn_outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=rnn_cell_fw,
        cell_bw=rnn_cell_bw,
        inputs=inputs,
        sequence_length=lengths,
        dtype=tf.float32
    )
    if return_sequences: return rnn_outputs #tf.concat(rnn_outputs, axis=-1)
    else: return tf.concat([state_fw, state_bw], axis=-1)


def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    #print(vu)
    #print(inputs)
    exps = tf.reshape(tf.exp(vu), shape=[-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def multi_label_classifier(x, params):
    kernel_initializer = None
    #if 'concept_embeddings' in params:
    #    concept_embeddings = init_embeddings(params.label_vocab, pretrained_embeddings=params.concept_embeddings)
    #    kernel_initializer = tf.constant_initializer(concept_embeddings)
    logits = tf.layers.dense(x, params.classes_count, activation=None, kernel_initializer=kernel_initializer)
    probs = tf.nn.softmax(logits)
    predictions = tf.argmax(probs, axis=-1)
    return logits, probs, predictions


def crf_predictions(logits, lengths, vocab_size):
    transition_matrix = tf.truncated_normal(shape=[vocab_size, vocab_size])
    #print logits
    #print lengths
    predictions, _ = tf.contrib.crf.crf_decode(logits, transition_matrix, lengths)
    return predictions, transition_matrix


def crf_loss(logits, labels, lengths, transition_matrix):
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, lengths, transition_matrix)
    return tf.reduce_mean(-log_likelihood)


def compose_estimator(mode, labels, logits, probs, predictions, params, lengths=None):
    # mode: PREDICT
    # ugly code, need to refactor crf part
    if 'CRF' in params and params.CRF:
        predictions, transition_matrix = crf_predictions(logits, lengths, params.classes_count)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_labels = rev_lookup(predictions, params.label_vocab_file)
        return tf.estimator.EstimatorSpec(mode, predictions={'class': predicted_labels, 'prob': probs})

    # mode: TRAIN
    label_ids = lookup(labels, params.label_vocab_file)
    # print label_ids.shape
    if 'CRF' in params and params.CRF:
        loss = crf_loss(logits, label_ids, lengths, transition_matrix)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(label_ids, logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(params.learning_rate) \
            .minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # mode: EVAL
    eval_metrics_ops = metrics(label_ids, predictions)
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def bilstm_model(features, labels, mode, params):
    token_vocab_file = params.token_vocab_file
    pretrained_embeddings_file = params.pretrained_embeddings_file
    classes_count = params.classes_count

    embedded_sequence = embed(features['entity'], token_vocab_file, pretrained_embeddings_file)

    encoding = rnn_layer(embedded_sequence, features['length'], mode,
                         params, keep_prob=.8, input_size=embedded_sequence.shape[-1])

    if 'attention_size' in params:
        encoding = attention(encoding, params.attention_size)

    #if 'auxilary_features' in features:
    #    encoding = tf.concat([encoding, features['auxilary_features']], axis=1)

    logits, probs, predictions = multi_label_classifier(encoding, params)

    return compose_estimator(mode, labels, logits, probs, predictions, params)


def elmo_model(features, labels, mode, params):
  classes_count = params.classes_count
  max_length = features['entity'].shape[-1]
  tokens = features['entity']
  lengths = features['length']
  embedded_sequence = elmo_embed(tokens, lengths)
  embedded_sequence = rnn_layer(embedded_sequence, lengths, mode, params,  keep_prob=0.9, input_size=1024, return_sequences=False)
  # embedded_sequence = tf.stack(embedded_sequence, axis=-1)
  # embedded_sequence = tf.reshape(embedded_sequence, shape=[-1, max_length, params.rnn_num_units*2])
  # embedded_sequence = attention(embedded_sequence, params.attention_size)
  logits, probs, predictions = multi_label_classifier(embedded_sequence, params)
  return compose_estimator(mode, labels, logits, probs, predictions, params, lengths)
