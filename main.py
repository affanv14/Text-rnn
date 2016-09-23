import utils
import os
from model import charrnn
from config import trainconfig, testconfig
import tensorflow as tf
from optparse import OptionParser
import cPickle


def main():
    parser = OptionParser()
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
                      default=False, help='no output in terminal')
    parser.add_option('-i', '--input', dest='input_file',
                      help='file for input')
    parser.add_option('-u', '--summary', dest='summary',
                      action='store_true', help='generate summary', default=False)
    parser.add_option('-s', '--save', dest='save',
                      help='specify save file(without extension)', default='checkpoint')
    parser.add_option('-g', '--generate', dest='generate',
                      help='generate data', action='store_true', default=False)
    parser.add_option('-d', '--debug', dest='debug',
                      action='store_true', default=False)
    (options, args) = parser.parse_args()

    if not options.generate:
        train(options=options)
    else:
        test(options)


def train(options):
    train_config = trainconfig()
    if options.debug:
        train_config.num_epochs = 1
    if options.input_file is None:
        raise ValueError("Input file not specified")
    input_file = options.input_file
    if not os.path.isfile(input_file):
        raise ValueError("Input file not found")
    x, y, char2idx, idx2char = utils.preprocess(input_file, train_config.batch_size,
                                                train_config.num_timesteps)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("charrnn", reuse=None):
            trainmodel = charrnn(train_config, len(char2idx), True)
        saver = tf.train.Saver()
        if not os.path.isfile('saves/' + options.save + '.pkt'):
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, 'saves/' + options.save + '.pkt')
        if options.summary:
            train_writer = tf.train.SummaryWriter(
                options.summary + '/train', sess.graph)
        state = trainmodel.init_state.eval()
        for i in range(train_config.num_epochs):
            for j in range(len(x)):
                feed_dict = {trainmodel.input_placeholder: x[j], trainmodel.target: y[j],
                             trainmodel.init_state: state}
                loss, state, output, _ = sess.run(
                    [trainmodel.loss, trainmodel.final_state, trainmodel.softmax_output,
                     trainmodel.train_op], feed_dict=feed_dict)
                if not options.quiet:
                    print 'epoch(%d/%d) loss=%f' % (i, train_config.num_epochs, loss)
            saver.save(sess, 'saves/' + options.save + '.pkt')
        with open('saves/vars.pkl', 'wb') as varfile:
            cPickle.dump((char2idx, idx2char), varfile)


def test(options):
    test_config = testconfig()
    if not os.path.isfile('saves/var.pkl'):
        assert ValueError('Var file not found.Run train first')
    with open('saves/vars.pkl', 'r') as varfile:
        (char2idx, idx2char) = cPickle.load(varfile)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("charrnn", reuse=None):
            testmodel = charrnn(test_config, len(char2idx), False)
        saver = tf.train.Saver()
        if not os.path.isfile('saves/' + options.save + '.pkt'):
            assert ValueError('checkpoint file not found run train first')
        else:
            saver.restore(sess, 'saves/' + options.save + '.pkt')

        chars = list(test_config.start_str)
        x = testmodel.sample(sess, test_config.num_chars,
                             [char2idx[i] for i in chars])
        print ''.join([idx2char[i] for i in x])
main()
