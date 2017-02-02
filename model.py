import numpy as np
import tensorflow as tf


class charrnn(object):

    def __init__(self, config, vocab_size, is_training,keep_prob,num_cells):

        self.num_timesteps = config.num_timesteps
        self.num_units = config.num_units
        self.batch_size = config.batch_size
        self.input_placeholder = tf.placeholder(
            tf.int32, [self.batch_size, self.num_timesteps])
        self.target = tf.placeholder(
            tf.int32, [self.batch_size, self.num_timesteps])
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_cells)
        with tf.variable_scope('embed'):
            embedding = tf.get_variable(
                "embedding", [vocab_size, self.num_units])
            embedded_inputs = tf.nn.embedding_lookup(
                embedding, self.input_placeholder)

        self.init_state = cell.zero_state(self.batch_size, tf.float32)
        rnn_output, self.final_state = tf.nn.dynamic_rnn(
            cell, embedded_inputs, dtype=tf.float32, initial_state=self.init_state)
        concat_output = tf.reshape(
            tf.concat(1, rnn_output), [-1, self.num_units])
        with tf.variable_scope('softmax'):
            w = tf.get_variable('w', shape=[self.num_units, vocab_size])
            b = tf.get_variable('b', shape=[vocab_size])
        self.softmax_output = tf.matmul(concat_output, w) + b
        if not is_training:
            return
        seqloss = tf.nn.seq2seq.sequence_loss_by_example([self.softmax_output], [
                                                         tf.reshape(self.target, [-1])],
                                                         [tf.ones([self.batch_size * self.num_timesteps])])
        self.loss = tf.reduce_mean(seqloss)

        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

    def sample(self, sess, num_chars, start_str):
        state =sess.run(self.init_state)
        for char in start_str:
            feed_dict = {self.input_placeholder: [[
                char]], self.init_state: state}
            str_output, state = sess.run(
                [self.softmax_output, self.final_state], feed_dict=feed_dict)
        strlist = list(start_str)
        str_input = start_str[-1]
        for i in range(num_chars):
            feed_dict = {self.input_placeholder: [[
                str_input]], self.init_state: state}
            str_output, state = sess.run(
                [self.softmax_output, self.final_state], feed_dict=feed_dict)
            str_input = np.argmax(np.squeeze(str_output))
            strlist.append(str_input)
        return strlist
