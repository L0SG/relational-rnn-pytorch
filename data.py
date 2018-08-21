import os
import torch
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.idx2count.append(1)
            self.word2idx[word] = len(self.idx2word) - 1
        else:
            self.idx2count[self.word2idx[word]] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        tokens_train = self.add_corpus(os.path.join(path, 'train.txt'))
        tokens_valid = self.add_corpus(os.path.join(path, 'valid.txt'))
        tokens_test = self.add_corpus(os.path.join(path, 'test.txt'))

        # sort the words by word frequency in descending order
        # this is for using adaptive softmax: it assumes that the most frequent word get index 0
        idx_argsorted = np.flip(np.argsort(self.dictionary.idx2count), axis=-1)

        # re-create given the sorted ones
        self.dictionary.idx2count = np.array(self.dictionary.idx2count)[idx_argsorted].tolist()
        self.dictionary.idx2word = np.array(self.dictionary.idx2word)[idx_argsorted].tolist()
        self.dictionary.word2idx = dict(zip(self.dictionary.idx2word,
                                            np.arange(len(self.dictionary.idx2word)).tolist()))

        self.train = self.tokenize(os.path.join(path, 'train.txt'), tokens_train)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), tokens_valid)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), tokens_test)

    def add_corpus(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        return tokens

    def tokenize(self, path, tokens):
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
