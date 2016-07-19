# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:25:45 2016

@author: whr94621
"""
import numpy as np
from collections import defaultdict

def _cosine(v1, v2):
    norm1 = np.dot(v1, v1)
    norm2 = np.dot(v2, v2)
    return np.dot(v1, v2) / norm1**0.5 / norm2**0.5

def get_pos_tag(tag_word):
    tag = tag_word.split('_')[-1]
    return tag


class Vocab(object):
    '''
    Vocab stores all of word embeddings as well as provides interfaces to fetch
    embeddings by IDs or Strings
    '''
    def __init__(self, tag_word_list, tag_word_dict, tag_word_vec,
                   tag_word_freq, word_dict, N, D):
        #id => tag_word
        self._tag_word_list = tag_word_list
        #tag_word => id
        self._tag_word_dict = tag_word_dict
        #id => vector
        self._tag_word_vec = tag_word_vec
        #id => freqeuncy
        self._tag_word_freq = tag_word_freq
        #word => tag_word set
        self._word_dict = word_dict

        self._D = D
        self._N = N

    def tag_word2id(self, word):
        return self._tag_word_dict[word]

    def id2tag_word(self, ID):
        return self._tag_word_list[ID]

    def get_vec(self, query):
        if not isinstance(query, int):
            query = self.tag_word2id(query)
        return self._tag_word_vec[query,:]

    def random_sampling(self, flag='id'):
        ID = np.random.randint(self._N)
        if flag == 'word':
            return self.id_to_word(ID)
        else:
            return ID

    def neighbor_word(self, query, num):
        if not isinstance(query, int):
            query = self.tag_word2id(query)
        score_cos = np.ascontiguousarray(np.empty([self._N]))
        v_q = self.get_vec(query)
        for i in xrange(self._N):
            v_i = self.get_vec(i)
            score_cos[i] = _cosine(v_q, v_i)
        best_cos = np.argsort(-score_cos)[1:num+1]
        return set(best_cos)

    def tag(self, word):
        try:
            tag_words = self._word_dict[word]
            if len(tag_words) > 0:
                return tag_words
        except KeyError:
            return None

    @classmethod
    def load(cls, f_vocab, f_vec, D, coding='utf8'):
        tag_word_list = []
        tag_word_dict = {}
        word_dict = defaultdict(set)
        tag_word_freq = []
        tag_word_vec = []


        with open(f_vocab, 'r') as fin:
            for i, line in enumerate(fin):
                line = line.strip().split()
                tag_word = line[0].decode(coding)
                tag_word_dict[tag_word] = i
                tag_word_freq.append(int(line[1]))
                tag_word_list.append(tag_word)
                word = tag_word.split('_')[0]
                word_dict[word].add(tag_word)

        N = i + 1

        with open(f_vec, 'rb') as fin:
            vec = fin.read()

        tag_word_vec = np.fromstring(vec,dtype=np.float)
        tag_word_vec = np.reshape(tag_word_vec, (-1,D))
        return cls(tag_word_list, tag_word_dict, tag_word_vec,
                   tag_word_freq, word_dict, N, D)



if __name__ == '__main__':
    vocab = Vocab.load('../data/input/vocab_100_5_100.txt',
    '../data/input/vector_100_5_100.bin',100)







