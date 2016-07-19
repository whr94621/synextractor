# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:02:18 2016

@author: whr94621
"""

import tensorflow as tf
from synonym import Synonym


##################
# mutable params #
##################
top = 15
conf1 = 'configure.json'
conf2 = 'configure_1.json'
cilin = '../data/cilin.txt'

# the script
g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    syn1 = Synonym.load(conf1)
with g2.as_default():
    syn2 = Synonym.load(conf2)

with open(cilin, 'r') as f, open('output.txt', 'w') as g:
    for line in f:
        line = line.strip().decode('utf8')
        words = line.split()[1:]
        tag = line.split()[0]
        more_syns = set(words)
        for word in words:

            with g1.as_default():
                syns1 = syn1.generate_synonyms(word, top)
            with g2.as_default():
                syns2 = syn2.generate_synonyms(word, top)

            if syns1 and syns2:
                syns_1 = syns1.values()[0]
                syns_2 = syns2.values()[0]
                syns_1 = [w.split('_')[0] for w in syns_1]
                syns_2 = [w.split('_')[0] for w in syns_2]
                syns = set(syns_1) & set(syns_2)
            more_syns = more_syns | syns
        new_line = '%s %s\n' %  (tag, ' '.join(more_syns))
        g.write(new_line.encode('utf8'))

