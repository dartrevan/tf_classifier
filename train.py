from spell_checker import SpellChecker
from data_loader import get_input_fn
from models import bilstm_model, elmo_model, lookup
from utils import load_config
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', '', '')
MAX_STEPS = 3688 * 2


def train(argv=None):
    params = load_config(FLAGS.config_file)
    spell_checker = None
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=params.save_chkpts_steps,
        model_dir=params.model_dir
    )
    estimator = tf.estimator.Estimator(bilstm_model, config=run_config, params=params)

    train_input_fn = get_input_fn(
            features=params.data['train']['features'],
            target=params.data['train']['target'],
            batch_size=64,
            train=True
        )
    #lookup_table = tf.contrib.lookup.index_table_from_file(
    #     vocabulary_file='/root/DATA/medical_processing_corpora/AskAPatient/normalization_plain_fold_5/token_vocab.txt',
    #    vocab_size=None,
    #    default_value=0
    #)
    #sess = tf.Session()
    #sess.run([ tf.global_variables_initializer(), tf.tables_initializer()])
    #features, target = train_input_fn()
    #label_ids = lookup_table.lookup(target)
    #token_ids = lookup_table.lookup(features['entity'])
    #print(sess.run(label_ids))
    #print(sess.run(token_ids))
    #print(sess.run(features))
    #exit()
    if True or 'eval_target' in params and 'eval_texts' in params:
        eval_input_fn = get_input_fn(
            features=params.data['test']['features'],
            target=params.data['test']['target'],
            batch_size=64,
            train=False
        )
        #sess = tf.Session()
        #sess.run([ tf.global_variables_initializer(), tf.tables_initializer()])
        #features, target = train_input_fn()
        #label_ids = lookup_table.lookup(target)
        #print(sess.run(label_ids))
        train_spec = tf.estimator.TrainSpec(train_input_fn, MAX_STEPS)
        eval_spec = tf.estimator.EvalSpec(eval_input_fn, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        estimator.train(train_input_fn, max_steps=MAX_STEPS)


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    tf.app.run(main=train)
