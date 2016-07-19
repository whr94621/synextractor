# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:20:23 2016

@author: whr94621
"""

import optparse as opt
import os
import sys
from data import build_data
import Model
import tensorflow as tf
import numpy as np
import time
import json
from itertools import islice
import signal as Signal

def _print_bar(i, N, message=''):
    """Print process bar
    """
    n = int(i * 20.0 / N)
    sys.stdout.write('\r')
    sys.stdout.write('\r%d%%' % int(i * 100.0 / N))
    sys.stdout.write('#' * n +' ' * (20-n) + '|' )
    sys.stdout.write(message)
    sys.stdout.flush()


def _generate_data(file, pos_file, dim, window, threads, min_count):

    """Run word2vec in python
    Train word2vec by google c-style word2vec.
    Vector file name is like 'vector_($dim)_($window)_($mini_count).txt'.
    Vocab file name is like 'vocab_($dim)_($window)_($mini_count).txt'.
    Both of two will be located at ../bin/

    Args:
        file: string, input raw corpus
        dim: int, dimension of the embedding
        window: int, half size of word window
        threads: int, threads to run word2vec
        mini_count: minimum count of words

    Return:
        flag: string, the flag specify the embedding feature.The flag format
              as _$(dim)_$(window)_$(mini_count)
    """


    data_dir = "../data"
    input_dir = "%s/input" % data_dir
    temp_dir = "%s/temp" % data_dir

    def my_signal_handler(signal, frame):
        if signal == Signal.SIGINT:
            os.system('rm %s/*%s*')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    flag = '_%d_%d_%d' % (dim, window, min_count)
    save_vector = flag.join(['%s/vector' % input_dir, '.txt'])
    save_vocab = flag.join(['%s/vocab' % input_dir,'.txt'])
    save_vector_bin = flag.join(['%s/vector' % input_dir,'.bin'])

    if not os.path.exists(save_vector) and \
        not os.path.exists(save_vector_bin):

        command = ("../bin/word2vec -train %s -output %s -cbow 0 -size "
                   "%d -window %d"
                   " -hs 1 -negative 0 -threads %d -min-count "
                   "%d -save-vocab %s") % \
                   (file, save_vector, dim, window, threads, min_count,
                    save_vocab)

        os.system(command)

    #convert a float list to numpy array
    def _list_to_ndarray(l):
        L = len(l)
        vec = np.empty(shape=L, dtype=np.float64)
        for i, e in enumerate(l):
            vec[i] = np.float64(e)
        return vec

    if not os.path.exists(save_vector_bin):
        with open(save_vector, 'r') as f, \
                open(save_vector_bin, 'wb')  as g:
                    title = f.readline()
                    n = int(title.strip().split()[0])
                    sys.stdout.write("\n...transform data embedding format\n")
                    for i, line in enumerate(f):
                        _print_bar(i+1,n)
                        line = line.strip().split()[1:]
                        vec = _list_to_ndarray(line)
                        g.write(vec.tostring())
                        sys.stdout.flush()

        os.system(" ".join(["rm", save_vector]))


    #generate train and test set
    if not os.path.exists("%s/train_y%s.txt" % (temp_dir, flag)):
        build_data(pos_file, save_vocab, save_vector_bin, flag, dim)

    return flag, input_dir


def _train(flag, input_shape=[-1,250], expand_layer=[3,20],
          layers=[250,50,1], learning_rate=0.1, momentum=0.8,
            n_epoch=100, block_bytes=3*50*8):
    """Train Synonym extraction  model

    Args:
        flag: string, format like _$(dim)_$(window)_$(min_count), identifier

        input_shape: list of int, specify the shape of placeholder

        expand_layer: list of int, specify the feature expand layer.
                      For example, if value is [3, 20], that means expand
                      feature from 3 to 20 with a 3*20 matrix

        layers: list of int, specify the number of nerons in the MLP.For
                example, if value is [250, 50, 1], that means input layer is
                batch*250, hidden layer is 250*50, output layer is 50*1.

    """
    with tf.variable_scope('var%s' % flag):

        data_dir = "%s/data" % os.path.dirname(os.getcwd())
        model_dir = "%s/model" % data_dir
        this_dir = "%s/%s" % (model_dir, flag)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(this_dir):
            os.mkdir(this_dir)

        f_train_x = flag.join(['../data/temp/train_X','.bin'])
        f_train_y = flag.join(['../data/temp/train_y','.txt'])
        f_test_x = flag.join(['../data/temp/test_X','.bin'])
        f_test_y = flag.join(['../data/temp/test_y','.txt'])

        placeholder_shape = list(input_shape)
        placeholder_shape[0] = None

        print '...building graph%s' % flag
        sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=4))
            #Build graph
        if  len(input_shape) == 2:
            model = Model.MLP(momentum=momentum,
                                  input_shape=placeholder_shape, layers=layers)
        elif len(input_shape) == 3:
            model = Model.MLPwL(momentum=momentum,
                            input_shape=placeholder_shape,
                            expand_layer=expand_layer, layers=layers)
        else:
            raise ValueError

        # Build operations
        x_placeholder = model.x_placeholder
        y_placeholder = model.y_placeholder
        pred = model.output(x_placeholder)
        loss = model.loss_function(pred, y_placeholder)
        train_op = model.train_op(loss)

        # Builde saver
        saver = tf.train.Saver()
        tf.add_to_collection('output', pred)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('input', x_placeholder)

        # Train model
        sess.run(tf.initialize_all_variables())
        print '...start training %s' % flag
        t0 = time.time()
        for epoch in xrange(n_epoch):
            gen = Model.data_generator(f_train_x, f_train_y, block_bytes)
            batch = list(islice(gen, 50))
            while(len(batch) > 0):
                batch_x = np.array([l[0] for l in batch])
                batch_x = np.reshape(batch_x, input_shape)
                batch_y = np.array([l[1] for l in batch])

                feed_dict = {x_placeholder:batch_x,
                         y_placeholder:batch_y}
                ls, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                batch = list(islice(gen, 50))

            t1 = time.time()
            message = ('epoch: %d, loss: %.5f, elapsed time: %.5f'
            % (epoch + 1, ls, t1 - t0))
            _print_bar(epoch + 1, n_epoch, message)

        # Print Model Report
        gen = Model.data_generator(f_test_x, f_test_y,
                        block_bytes=block_bytes)
        batch = list(islice(gen, 50000))
        batch_x = np.array([l[0] for l in batch])
        batch_x = np.reshape(batch_x, input_shape)
        batch_y = np.array([l[1] for l in batch])
        x = sess.run(pred, {x_placeholder:batch_x})
        x[x > 0] = 1
        x[x < 0] = -1
        precision, recall, f1 = Model.metric(x, batch_y)
        print '\n'
        print 'precision: %f, recall: %f, f1 score: %f' % (precision,
                                                           recall, f1)

        # Saving graph and architecture and variables
        graph_def = sess.graph_def
        saver.save(sess, flag.join(['%s/model' % this_dir,'']))
        saver_def = saver.saver_def

        filename = flag.join(['%s/model' % this_dir,'.meta'])
        collection_list = ['output', 'loss', 'input']
        tf.train.export_meta_graph(filename=filename,
                               graph_def=graph_def,
                               saver_def=saver_def,
                               collection_list=collection_list)
        os.system('rm -rf ../data/temp')
        return model_dir

def arg_parse():
    """argument parser
    """


    usage = "usage: %prog [option] arg1 arg2"
    Opt = opt.OptionParser(usage=usage)
    # necessary arguments

    Opt.add_option("-d", "--dim", action="store", type="int",
                             dest="dim", help="The dimension of embedding.",
                             default=50)

    Opt.add_option("-w", "--window", action="store", type="int",
                             dest="window",
                             help="The half length of word window.",
                             default=5)

    Opt.add_option("--min_count", action="store", type="int",
                             dest="min_count", help="Minimum count.",
                             default=20)

    #GENERATE DATA OPTIONS
    Opt_data = opt.OptionGroup(Opt, "GENERATE DATA OPTIONS",
        "Options below are arguments to generate data.")

    Opt_train = opt.OptionGroup(Opt, "TRAIN MODEL OPTIONS",
        "Options below are arguments to train model.")

    Opt_clean = opt.OptionGroup(Opt, "CLEAN THE DATA AND MODEL",
        "Options below are arguments to clean the specific data and model")

    Opt.add_option_group(Opt_data)
    Opt.add_option_group(Opt_train)
    Opt.add_option_group(Opt_clean)

    # Opt_data arguments
    Opt_data.add_option("--generate_data", action="store_true",
                   dest="is_generate_data",
                   help='''option to generate train and test data.
                           See GENERATE DATA OPTIONS for details''')

    Opt_data.add_option("-f", "--file", action="store", dest="file",
                   help='''Input file, this option could be raw corpus,
                           positive synonym pairs, etc. See more details
                           in each option group.''')

    Opt_data.add_option("-p", "--postive_data", action="store",
                        type="string", dest="postive_data",
                        help="Synonym pairs used as postive_data.")

    Opt_data.add_option("-t","--threads", action="store", type="int",
                             dest="threads",
                             help="Number of threads to run word2vec.",
                             default=1)



    # Opt_train arguments

    Opt_train.add_option("--train_model", action="store_true",
                   dest="is_train_model",
                   help='''option to train synonym extraction model.
                           See MODEL TRAINING OPTIONS for details
                        '''
                   )

    Opt_train.add_option('-o','--output', action="store", type="string",
                         dest="output", help="Path of configure file.")

    Opt_train.add_option('--npoch', action="store", type="int",
                         dest="npoch", help="Number of epoches")

    # Opt_clean arguments
    Opt_clean.add_option('--clean', action="store_true",dest="is_clean",
                         help="Clean specific data and model.")

    return Opt


def main():
    Opt = arg_parse()
    options, args = Opt.parse_args()
    w_dir = os.path.dirname(os.getcwd())
    input_dir = None
    model_dir = None
    flag = None

    # Generate data
    if options.is_generate_data:

        flag, input_dir = _generate_data(options.file, options.postive_data,
                                         options.dim, options.window,
                                         options.threads, options.min_count)
    # Train model
    if options.is_train_model:
        if not flag:
            flag = '_%d_%d_%d' % (options.dim, options.window,
                                options.min_count)
        if not os.path.exists('%s/data/model/%s' % (w_dir, flag)):
            if os.path.exists('../data/temp/train_X%s.bin' % flag) and \
                os.path.exists('../data/temp/train_y%s.txt' % flag) and \
                os.path.exists('../data/temp/test_X%s.bin' % flag) and \
                os.path.exists('../data/temp/test_y%s.txt' % flag):
                    d = options.dim
                    model_dir = _train(flag, input_shape=[-1, 5*d],
                                       expand_layer=None,
                                       layers=[5*d,int(0.67 * 5 * d),1],
                                   learning_rate=0.1, momentum=0.9,
                                   n_epoch=options.npoch, block_bytes=5*d*8)
            else:
                print 'Lack of training data!'
        else:
            print 'Model has been trained!'


    # Saving model configure
    if flag and model_dir:
        configure = {'flag':flag, 'input_shape':[-1, options.dim],
                 'input_dir':input_dir, 'model_dir':model_dir}
        with open(options.output,'w') as f:
            f.write(json.dumps(configure))

    # Clean model and data
    if options.is_clean:
        flag = '_%d_%d_%d' % (options.dim, options.window,
                                options.min_count)
        os.system('rm -rf ../data/input/*%s*' % flag)
        os.system('rm -rf ../data/temp')
        os.system('rm -rf ../data/model/%s/' % flag)



if __name__ == '__main__':
    main()




