# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:12:31 2016

@author: whr94621
"""
import tensorflow as tf
import json
from words import Vocab
from data import generate_feature
import numpy as np
import optparse as opt
import sys

def _print_bar(i, N, message=''):
    """Print process bar
    """
    n = int(i * 50.0 / N)
    sys.stdout.write('\r')
    sys.stdout.write('\r%d%%' % int(i * 100.0 / N))
    sys.stdout.write('#' * n +' ' * (50-n) + '|' )
    sys.stdout.write(message)
    sys.stdout.flush()

class Synonym:

    def __init__(self, vocab_file, vec_file, graph_data, variable_data,
                 flag, input_shape=[-1, 250]):

        dim = int(flag.split('_')[1])
        with tf.variable_scope('var%s' % flag):
            saver = tf.train.import_meta_graph(graph_data)

            sess = tf.Session()
            saver.restore(sess, variable_data)
            op_output = tf.get_collection('output')[0]
            x_input = tf.get_collection('input')[0]
            vocab = Vocab.load(vocab_file, vec_file, dim)

        self.sess = sess
        self.output = op_output
        self.x_input = x_input
        self.vocab = vocab
        self.input_shape = input_shape
        self.saver = saver
        self.dim = dim

    def is_synonym(self, x):
        pred = self.sess.run(self.output, feed_dict={self.x_input:x})
        pred[pred > 0 ] = 1
        pred[pred < 0] = 0
        return pred

    def get_syn(self, tag_word, top=20):

        w_id = self.vocab.tag_word2id(tag_word)
        w_v = self.vocab.get_vec(w_id)
        w_tag = tag_word.split('_')[-1]
        candidate = self.vocab.neighbor_word(w_id, top)
        candidate_v = [self.vocab.get_vec(i) for i in candidate]
        syn = []
        for idx, v in zip(candidate, candidate_v):
            in_v = generate_feature(w_v, v)
            in_v = np.reshape(in_v, self.input_shape)
            result = self.sess.run(self.output, feed_dict={self.x_input:in_v})
            if result > 0:
                syn.append(self.vocab.id2tag_word(idx))
        if len(syn) > 0:
            syn = [word for word in syn if w_tag == word.split('_')[-1]]
        syn = set(syn)
        return syn

    def tag(self, w):
        return self.vocab.tag(w)

    def generate_synonyms(self, word, top=20):
        result = {}
        tag_words = self.tag(word)
        if tag_words:
            for tag_word in tag_words:
                tag = tag_word.split('_')[-1]
                tag_syns = self.get_syn(tag_word, top)
                if len(tag_syns) > 0:
                    result[tag] = tag_syns
            return result
        else:
            return None


    @classmethod
    def load(cls, configure_file):
        with open(configure_file, 'r') as f:
            configure = f.read()

        configure = json.loads(configure)
        flag = configure['flag']
        input_dir = configure['input_dir']
        model_dir = configure['model_dir']
        input_shape = configure['input_shape']

        vocab_file = '%s/vocab%s.txt' % (input_dir, flag)
        vec_file = '%s/vector%s.bin' % (input_dir, flag)
        graph_data = '%s/%s/model%s.meta' % (model_dir, flag, flag)
        varaible_data = '%s/%s/model%s' % (model_dir, flag, flag)

        return cls(vocab_file=vocab_file, vec_file=vec_file,
                  graph_data=graph_data, variable_data=varaible_data,
                  flag=flag, input_shape=input_shape)

def generate_thesaurus(model1, model2, file, output, top=10):


    with open(file, 'r') as f, \
            open (output, 'w') as g:
                for idx, _ in enumerate(f):
                    sys.stdout.write('\r%d lines' % idx)
                    sys.stdout.flush()
                sys.stdout.write('\n')
                N = idx + 1
                f.seek(0)
                g1 = tf.Graph()
                g2 = tf.Graph()
                with g1.as_default():
                    syn1 = Synonym.load('configure_1.json')
                with g2.as_default():
                    syn2 = Synonym.load('configure.json')
                for idx, line in enumerate(f):
                    word = line.decode('utf8').strip().split()[0]
                    syns1 = syn1.get_syn(word, 20)
                    syns2 = syn2.get_syn(word, 20)
                    syns = syns1 & syns2
                    if len(syns) > 0:
                        s = [w.encode('utf8') for w in syns]
                        g.write('%s\t%s\n' %
                            (word.encode('utf8'), ' '.join(s)))
                    else:
                        g.write('%s\n' % word.encode('utf8'))

                    message = 'building %d words' % (idx+1)
                    _print_bar(idx+1, N, message=message)


def generate_synonyms(model1, model2, word, top=20):
    synonyms = {}
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        syn1 = Synonym.load(model1)
    with g2.as_default():
        syn2 = Synonym.load(model2)

    word = word.decode('utf8').strip()
    with g1.as_default():
        syns1 = syn1.generate_synonyms(word, top)
    with g2.as_default():
        syns2 = syn2.generate_synonyms(word, top)
    tag_words = syn2.tag(word)
    if syns1 and syns2 and tag_words:
        if len(syns1) > 0 and len(syns2) > 0:
            for tag_word in tag_words:
                tag = tag_word.split('_')[-1]
                ss = syns1[tag] & syns2[tag]
                if ss:
                    synonyms[tag] = ss
        return synonyms
    else:
        return None


def main():
    usage = "usage: %prog [option] arg1 arg2"
    Opt = opt.OptionParser(usage=usage)
    # necessary arguments
    Opt.add_option("--m1", action="store", dest='m1',
                   help='''the path of model I
                        ''')

    Opt.add_option("--m2", action="store", dest='m2',
                   help='''the path of model II
                        ''')

    Opt.add_option("-n","--num",action="store",type="int", dest='n',
                   help='''at most how many synoyms should we return.
                           10 and above is recommended.
                        ''')

    Opt_syn = opt.OptionGroup(Opt, "GENERATE SYNONYM GIVEN A SINGLE WORD")
    Opt_the = opt.OptionGroup(Opt, "GENERATE THESARUSE GIVEN A LIST OF WORD")

    #Options of Opt_syn
    Opt_syn.add_option("--synonym", action="store_true", dest="synonym",
                   help='''given a word, return its synonyms by POS
                        ''')

    Opt_syn.add_option("-w", "--word", action="store", type="string",
                   dest='word',
                   help='''a word without pos tag
                        ''')

    #Options of Opt_the
    Opt_the.add_option("--thesaurus", action="store_true", dest="thesaurus",
                   help='''given a list of pos tagged words, return
                           their synonyms by lines
                        ''')

    Opt_the.add_option("-f", "--file", action="store", type="string",
                       dest='file',
                       help='''a list of words with pos tag.
                               e.g., apple_n
                            ''')

    Opt_the.add_option("-o", "--output", action="store", type="string",
                       dest='output',
                       help='''path of the thesaurus
                            ''')

    Opt.add_option_group(Opt_syn)
    Opt.add_option_group(Opt_the)



    options, args = Opt.parse_args()


    if options.synonym and options.thesaurus:
        sys.stderr('can only choose one mode at one time.\n')
    elif options.synonym:
        syns = generate_synonyms(options.m1, options.m2, options.word,
                             options.n)
        if syns:
            if len(syns) > 0:
                for tag, words in syns.iteritems():
                    print tag + ':'
                    for w in words:
                        print w
            else:
                print 'No Synonyms!'
        else:
            print 'Unknown word!'

    elif options.thesaurus:
        generate_thesaurus(options.m1, options.m2, options.file,
                           options.output, options.n)

if __name__ == '__main__':
    main()
