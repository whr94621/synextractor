# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:07:42 2016

@author: whr94621
"""
from words import Vocab, get_pos_tag
import numpy as np

def generate_feature(v1, v2):
    v =  np.concatenate((v1, v2, np.abs(v1 - v2), v1 + v2, v1*v2))
    return v

def _data_sampling(vocab, f_pos, size):
    data_queue = []
    # Read in positive pairs
    with open(f_pos, 'r') as f:
        for line in f:
            w1, w2 = line.decode('utf8').strip().split()
            data_queue.append((vocab.tag_word2id(w1),
                               vocab.tag_word2id(w2), 1.0))
    # Generate negative pairs
    for i in xrange(size):
        id1 = vocab.random_sampling()
        id2 = vocab.random_sampling()
        data_queue.append((id1, id2, -1.0))

    #shuffle the data stream and yield
    np.random.shuffle(data_queue)
    for id1, id2, label in data_queue:
        v1 = vocab.get_vec(id1)
        v2 = vocab.get_vec(id2)
        v = generate_feature(v1, v2)
        v = v.tostring()
        yield v, label



def build_data(f_pos, f_vocab ,f_vec, flag, dim, size=600000, test_rate=0.33):

    f_train_x = flag.join(['../data/temp/train_X','.bin'])
    f_train_y = flag.join(['../data/temp/train_y','.txt'])
    f_test_x = flag.join(['../data/temp/test_X','.bin'])
    f_test_y = flag.join(['../data/temp/test_y','.txt'])

    print '...loading word embedding'
    vocab = Vocab.load(f_vocab, f_vec, dim)

    print '...POS tagging positive file'
    pos_file = '../data/input/pos.txt'
    with open(f_pos, 'r') as f, open(pos_file, 'w') as g:
        for line in f:
            line = line.decode('utf8').strip().split()
            tag_w_1s = vocab.tag(line[0])
            tag_w_2s = vocab.tag(line[1])
            if tag_w_1s and tag_w_2s:
                for tag_w_1 in tag_w_1s:
                    for tag_w_2 in tag_w_2s:
                        if tag_w_1 and tag_w_2:
                            if get_pos_tag(tag_w_1) == get_pos_tag(tag_w_2):
                                g.write(' '.join([tag_w_1.encode('utf8'),
                                              tag_w_2.encode('utf8')]))
                                g.write('\n')

    print 'positive file locates in %s' % pos_file

    print '...generating %s training and testing file' % flag
    print 'train_X file locates in %s' % f_train_x
    print 'train_y file locates in %s' % f_train_y
    print 'test_X file locates in %s' % f_test_x
    print 'test_y file locates in %s' % f_test_y
    flag = flag.split('_')[-1]
    with open(f_train_x, 'wb') as fout1, open(f_train_y, 'w') as fout2, \
        open(f_test_x,'wb') as fout3, open(f_test_y, 'w') as fout4:
            for v, label in _data_sampling(vocab, pos_file, size):
                i = np.random.random()
                if i > test_rate:
                    fout1.write(v)
                    fout2.write(str(label) + '\n')
                else:
                    fout3.write(v)
                    fout4.write(str(label) + '\n')


