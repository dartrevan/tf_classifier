from data_loader import get_input_fn
from models import bilstm_model, elmo_model
from utils import load_config
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', '', '')
flags.DEFINE_string('save_to', '', '')
flags.DEFINE_string('path_to_texts', None, '')
flags.DEFINE_string('path_to_lengths', None, '')
flags.DEFINE_string('path_to_afeatures', None, '')


def predict(arv=None):
    params = load_config(FLAGS.config_file)
    run_config = tf.estimator.RunConfig(
        model_dir=params.model_dir
    )
    input_fn = get_input_fn(
            features=params.data['test']['features'],
            target=params.data['test']['target'],
            batch_size=32,
            train=False
    )
    estimator = tf.estimator.Estimator(bilstm_model, config=run_config, params=params)
    predictions = estimator.predict(input_fn=input_fn)
    with open(FLAGS.save_to, 'w', encoding='latin-1') as output_stream:
        for prediction in predictions:
            output_stream.write(str(prediction['class'])[2:-1] + '\n')


if __name__ == '__main__':
    tf.app.run(main=predict)
