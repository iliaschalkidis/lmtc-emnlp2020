from typing import List
import os
import numpy as np

from data import VECTORS_DIR
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer


class Tokenizer(object):

    def __init__(self):
        pass

    def tokenize(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        raise NotImplementedError


class BERTTokenizer(Tokenizer):

    def __init__(self, bert_version):
        super().__init__()
        if 'roberta' in bert_version:
            self.tokenizer = RobertaTokenizer.from_pretrained(bert_version)
        elif 'scibert' in bert_version:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_version)

    def tokenize(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        word_inputs = np.zeros((len(sequences), max_sequence_size,), dtype=np.int32)
        for i, seq in enumerate(sequences):
            word_inputs[i] = self.tokenizer.encode(seq, max_length=max_sequence_size, pad_to_max_length=max_sequence_size)

        return word_inputs


class ELMoTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def tokenize(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        word_inputs = []
        # Encode ELMo embeddings
        for i, tokens in enumerate(sequences):
            sequence = ' '.join([token for token in tokens[:max_sequence_size]])
            if len(tokens) < max_sequence_size:
                sequence = sequence + ' ' + ' '.join(['#' for i in range(max_sequence_size - len(tokens))])
            word_inputs.append([sequence])

        return np.asarray(word_inputs)


class W2VTokenizer(Tokenizer):

    def __init__(self, w2v_model='glove.6B.200d.txt'):
        super().__init__()
        self.w2v_model = w2v_model

        self.word_indices = {'PAD': 0}
        count = 1
        with open(os.path.join(VECTORS_DIR, w2v_model)) as file:
            for line in file.readlines()[1:]:
                self.word_indices[line.split()[0]] = count
                count += 1

    def tokenize(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        """
        Produce W2V indices for each token in the list of tokens
        :param sequences: list of lists of tokens
        :param max_sequence_size: maximum padding
        """

        word_inputs = np.zeros((len(sequences), max_sequence_size, ), dtype=np.int32)

        for i, sentence in enumerate(sequences):
            for j, token in enumerate(sentence[:max_sequence_size]):
                if token.lower() in self.word_indices:
                    word_inputs[i][j] = self.word_indices[token.lower()]
                else:
                    word_inputs[i][j] = self.word_indices['unknown']
        
        return word_inputs
