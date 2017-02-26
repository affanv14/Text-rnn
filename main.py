import utils
import os
from model import charrnn
import tensorflow as tf
from optparse import OptionParser
import cPickle
from nltk import word_tokenize
import numpy as np


def main():
    parser = OptionParser()
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
                      default=False, help='no output in terminal')
    parser.add_option('-i', '--input', dest='input_file',
                      help='file for input')
    parser.add_option('-n', '--new', dest='new',
                      action='store_true', default=False)
    parser.add_option('-x', '--summary', dest='summary',
                      action='store_true', help='generate summary', default=False)
    parser.add_option('-c', '--checkpoint', dest='save',
                      help='specify save file(without extension)', default='checkpoint')
    parser.add_option('-g', '--generate', dest='generate',
                      help='generate data', action='store_true', default=False)
    parser.add_option('-d', '--debug', dest='debug',
                      action='store_true', default=False)
    parser.add_option('-p', '--prob', dest='keep_prob',
                      help='dropout probability', default=0.8, type=float)
    parser.add_option('-l', '--layers', dest='num_layers',
                      help='number of layers in rnn', default=2, type=int)
    parser.add_option('-w', '--word', dest='word', action='store_true',
                      default=False, help='learn and generate words')
    parser.add_option('-u', '--numunits', dest='num_units', default=128,
                      help='number of units of rnn', type=int)
    parser.add_option('-t', '--timesteps', dest='num_timesteps', default=50,
                      help='number of timesteps of the rnn', type=int)
    parser.add_option('-s', '--sample', dest='start_str', default='the',
                      help='sample string for generating')
    parser.add_option('-e', '--epochs', dest='num_epochs', default=20,
                      help='number of epochs', type=int)
    parser.add_option('-b', '--batchsize', dest='batch_size',
                      default=64, help='size of a batch', type=int)
    parser.add_option('-z', '--numchars', dest='num_chars', default=500,
                      help='number of characters to generate', type=int)

    (options, args) = parser.parse_args()

    if not options.generate:
        train(options=options)
    else:
        test(options)


def train(options):
    if options.debug:
        options.num_epochs = 1
    if options.input_file is None:
        raise ValueError("Input file not specified")
    input_file = options.input_file
    if not os.path.isfile(input_file):
        raise ValueError("Input file not found")
    x, y, element2idx, idx2element = utils.preprocess(input_file, options.batch_size,
                                                      options.num_timesteps, options.word)

    with tf.Graph().as_default(), tf.Session() as sess:

        if (os.path.isfile('saves/' + options.save) and os.path.isfile('saves/vars.pkl')) and not options.new:
            with open('saves/vars.pkl', 'r') as varfile:
                (_, __, wordgen) = cPickle.load(varfile)
            if wordgen == options.word:
                with tf.variable_scope("charrnn", reuse=None):
                    trainmodel = charrnn(options, len(
                        element2idx), True)
                saver = tf.train.Saver()
                saver.restore(sess, 'saves/' + options.save + '.pkt')
            else:
                raise ValueError(
                    'Save file contains indexes for different type of elements.\n'
                    'Use clear option to overwrite or specify different save path')
        else:
            with tf.variable_scope("charrnn", reuse=None):
                trainmodel = charrnn(options, len(
                    element2idx), True)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

        if options.summary:
            train_writer = tf.train.SummaryWriter(
                options.summary + '/train', sess.graph)
        state = sess.run(trainmodel.init_state)
        with open('saves/vars.pkl', 'wb') as varfile:
            cPickle.dump((element2idx, idx2element, options.word), varfile)
        for i in range(options.num_epochs):
            for j in range(len(x)):
                feed_dict = {trainmodel.input_placeholder: x[j], trainmodel.target: y[j],
                             trainmodel.init_state: state}
                loss, state,  _ = sess.run(
                    [trainmodel.loss_value, trainmodel.final_state,
                     trainmodel.train_op], feed_dict=feed_dict)
                if not options.quiet:
                    print('epoch(%d/%d) loss=%f' %
                          (i + 1, options.num_epochs, loss))
                if (j % (len(x) / 4) == 0):
                    path = saver.save(sess, 'saves/' + options.save + '.pkt')


def test(options):
    if not os.path.isfile('saves/vars.pkl'):
        raise ValueError('Var file not found.Run train first')
    with open('saves/vars.pkl', 'r') as varfile:
        (element2idx, idx2element, wordgen) = cPickle.load(varfile)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("charrnn", reuse=None):
            testmodel = charrnn(options, len(
                element2idx), False)
        saver = tf.train.Saver()
        saver.restore(sess, 'saves/' + options.save + '.pkt')
        if wordgen:
            elements = word_tokenize(options.start_str)
        else:
            elements = list(options.start_str)
        elements = [element2idx[i] for i in elements]
        state = sess.run(testmodel.cell.zero_state(1, tf.float32))
        for char in elements:
            feed_dict = {testmodel.input_placeholder: [[
                char]], testmodel.init_state: state}
            str_output, state = sess.run(
                [testmodel.softmax_output, testmodel.final_state], feed_dict=feed_dict)
        strlist = list(elements)
        str_input = np.random.choice(
            range(testmodel.vocab_size), p=np.squeeze(str_output))
        for i in range(options.num_chars):
            feed_dict = {testmodel.input_placeholder: [[
                str_input]], testmodel.init_state: state}
            str_output, state = sess.run(
                [testmodel.softmax_output, testmodel.final_state], feed_dict=feed_dict)
            str_input = np.random.choice(
                range(testmodel.vocab_size), p=np.squeeze(str_output))
            strlist.append(str_input)
        if wordgen:
            print(' '.join([idx2element[i] for i in strlist]))
        else:
            print(''.join([idx2element[i] for i in strlist]))


main()
