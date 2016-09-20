import utils
import os
from model import charrnn
import config
import tensorflow as tf


def main():
    input_file = 'text.txt'
    if not os.path.isfile(input_file):
        print "error:file not valid"
        return
    idxlist, charlist, char2idx, idx2char = utils.preprocess(input_file)
    x = idxlist[:-(len(idxlist) % config.num_timesteps)]
    y = x[1:]
    y.append(char2idx['.'])
    num_batches = len(idxlist) / config.num_timesteps
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("charrnn", reuse=None):
            trainmodel = charrnn(config, len(char2idx), True)
        with tf.variable_scope("charrnn", reuse=True):
            testmodel = charrnn(config, len(char2idx), False)

        saver = tf.train.Saver()
        if not os.path.isfile('saves/checkpoint.pkt'):
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, 'saves/checkpoint.pkt')
        train_writer = tf.train.SummaryWriter('summary/train', sess.graph)
        state = trainmodel.init_state.eval()
        for i in range(config.num_epochs):
            for j in range(num_batches):
                feed_dict = {trainmodel.input_placeholder: x[config.num_timesteps * j:config.num_timesteps * (
                    j + 1)], trainmodel.target: y[config.num_timesteps * j:config.num_timesteps * (j + 1)], trainmodel.init_state: state}
                loss, state, output, _ = sess.run(
                    [trainmodel.loss, trainmodel.final_state, trainmodel.softmax_output, trainmodel.train_op], feed_dict=feed_dict)
                print 'epoch %d loss=%f' % (i, loss)
            saver.save(sess, 'saves/checkpoint.pkt')
        chars = list('merchants ')
        x = testmodel.sample(sess, 200, [char2idx[i] for i in chars])
        print ''.join([idx2char[i] for i in x])
main()
