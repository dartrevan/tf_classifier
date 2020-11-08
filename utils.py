from tensorflow.contrib.training import HParams
from gensim.models import KeyedVectors
import nltk.tokenize as tokenizers
from spell_checker import SpellChecker
import numpy as np
import argparse
import codecs
import yaml
from nltk.stem import WordNetLemmatizer

spell_checker = SpellChecker()
lemmatizer = WordNetLemmatizer()

t_functions = {
    'word': tokenizers.word_tokenize,
    'wordpunct': tokenizers.wordpunct_tokenize,
    'none': lambda t: [t]
}


def pad_sequence(sequence, max_len=60, padding_value='<PAD>'):
    padding_size = max(max_len - len(sequence), 0)
    return sequence[:max_len] + [padding_value] * padding_size


def _preprocess_bunch(text, max_len=None, padding_value='<PAD>', wordlevel=True, check_spelling=True, sep='<SEP>', bunch_size=None):
    processed_texts = text
    if not isinstance(processed_texts, str): processed_texts = processed_texts.decode('utf-8', errors='ignore')
    processed_texts = processed_texts.lower().split(sep)
    if wordlevel:
        processed_texts = map(tokenizers.word_tokenize, processed_texts)
        if check_spelling:
            processed_texts = map(spell_checker.correct_sequence, processed_texts)
    else:
        processed_texts = map(list, processed_texts)
    if bunch_size is not None:
        processed_texts = pad_sequence(processed_texts, max_len=bunch_size, padding_value=[])
    if max_len is not None:
        processed_texts = map(lambda text: pad_sequence(
            text,
            max_len=max_len,
            padding_value=padding_value
        ), processed_texts)
    processed_texts = [[token.encode('utf-8') for token in text] for text in processed_texts]
    return [processed_texts]


def _preprocess(text, max_len=None, padding_value='<PAD>', wordlevel=True, check_spelling=False):
    processed_text = text
    if not isinstance(processed_text, str): processed_text = processed_text.decode('utf-8', errors='ignore')
    processed_text = processed_text.lower()
    if wordlevel:
        processed_text = tokenizers.word_tokenize(processed_text)
        if check_spelling:
            processed_text = spell_checker.correct_sequence(processed_text)
        #processed_text = [lemmatizer.lemmatize(token) for token in processed_text]
    else:
        processed_text = list(processed_text)

    if max_len is not None:
        processed_text = pad_sequence(
            processed_text,
            max_len=max_len,
            padding_value=padding_value
        )
    processed_text = [token.encode('utf-8') for token in processed_text]
    return [processed_text]


def build_vocab(texts, output_stream, tokenize_f=tokenizers.word_tokenize):
    vocab = set()
    for text in texts:
        for token in tokenize_f(text.lower()):
            #token = lemmatizer.lemmatize(token)
            vocab.add(token.strip('\n'))
    output_stream.write(u'<PAD>\n')
    output_stream.write(u'<UNKNOWN>\n')
    output_stream.write(u'\n'.join(vocab))


def count_lines(fname):
    with codecs.open(fname) as input_stream:
        for i, _ in enumerate(input_stream, start=1):
            pass
        return i


def init_embeddings(path_to_vocab, embedding_dim=200, pretrained_embeddings=None, normalize=True, binary=False):
    with codecs.open(path_to_vocab, encoding='utf-8') as input_stream:
        word2idx = {token.strip(): i for i, token in enumerate(input_stream)}
    embeddings = {}
    if pretrained_embeddings is not None:
        embeddings = KeyedVectors.load_word2vec_format(
            pretrained_embeddings,
            binary=binary,
            encoding='utf-8'
        )
        embedding_dim = embeddings.syn0.shape[1]

    pretrained = np.random.randn(len(word2idx), embedding_dim)/10
    invocab = 0
    for token, token_idx in word2idx.items():
        if token in embeddings:
            invocab += 1
            pretrained[token_idx] = embeddings[token]
    print("Pretrained embeddings found for", invocab/len(word2idx))
    #if normalize: pretrained /= np.linalg.norm(pretrained, axis=0)
    #print(pretrained.sum())
    return pretrained


def load_config(config_file):
    with codecs.open(config_file, encoding='utf-8') as input_stream:
        params = yaml.load(input_stream)
    hparams = HParams(**params)
    return hparams


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser("""Building vocab""")
    arg_parser.add_argument('--build_vocab', help='', action='store_true')
    arg_parser.add_argument('--indata', help='', nargs='+')
    arg_parser.add_argument('--saveto', help='')
    arg_parser.add_argument('--tokenizer', help='', default='word')
    args = arg_parser.parse_args()

    if args.build_vocab:
        tokenize_f = t_functions[args.tokenizer]
        lines = []
        for dtf in args.indata:
            with codecs.open(dtf, encoding='latin-1') as input_stream:
                for line in input_stream:
                    lines.append(line)
        with codecs.open(args.saveto, 'w', encoding='utf-8') as output_stream:
            build_vocab(lines, output_stream, tokenize_f=tokenize_f)
