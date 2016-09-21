import utils
import os
from model import charrnn
from config import trainconfig, testconfig
import tensorflow as tf


def main():
    train_config = trainconfig()
    test_config = testconfig()
    input_file = 'text.txt'
    if not os.path.isfile(input_file):
        print "error:file not valid"
        return
    x, y, char2idx, idx2char = utils.preprocess(input_file, train_config.batch_size,
                                                train_config.num_timesteps)

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("charrnn", reuse=None):
            trainmodel = charrnn(train_config, len(char2idx), True)
        with tf.variable_scope("charrnn", reuse=True):
            testmodel = charrnn(test_config, len(char2idx), False)

        saver = tf.train.Saver()
        if not os.path.isfile('saves/checkpoint.pkt'):
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, 'saves/checkpoint.pkt')
        train_writer = tf.train.SummaryWriter('summary/train', sess.graph)
        state = trainmodel.init_state.eval()
        for i in range(train_config.num_epochs):
            for j in range(len(x)):
                feed_dict = {trainmodel.input_placeholder: x[j], trainmodel.target: y[j],
                             trainmodel.init_state: state}
                loss, state, output, _ = sess.run(
                    [trainmodel.loss, trainmodel.final_state, trainmodel.softmax_output,
                     trainmodel.train_op], feed_dict=feed_dict)
                print 'epoch %d loss=%f' % (i, loss)
            saver.save(sess, 'saves/checkpoint.pkt')
        chars = list(test_config.start_str)
        x = testmodel.sample(sess, test_config.num_chars,
                             [char2idx[i] for i in chars])
        print ''.join([idx2char[i] for i in x])
main()
