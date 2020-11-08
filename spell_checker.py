import collections
import codecs


class SpellChecker(object):

    def __init__(self, spell_checker_vocab='/root/DATA/medical_processing_corpora/AskAPatient/normalization_plain_fold_5/token_vocab.txt'):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.model = collections.defaultdict(int)
        with codecs.open(spell_checker_vocab, encoding='utf-8') as input_stream:
            for line in input_stream:
                #token, occurrence = line.split('\t')
                token, occurrence = line.strip(), 1
                self.model[token] = occurrence

    def edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts = [a + c + b for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.model)

    def known(self, words):
        return set(w for w in words if w in self.model)

    def correct_word(self, word):
        word = word.lower()
        candidates = self.known([word]) or \
                     self.known(self.edits1(word)) or \
                     [word] if word.isalpha() else [word]
        return max(candidates, key=self.model.get)

    def correct_sequence(self, text):
        return [self.correct_word(token) for token in text]
