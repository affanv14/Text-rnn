import tensorflow as tf


class charrnn(object):

    def __init__(self, options, vocab_size, is_training):

        if(is_training):
            self.num_timesteps = options.num_timesteps
            self.batch_size = options.batch_size
            self.keep_probs = options.keep_prob
        else:
            self.num_timesteps = 1
            self.batch_size = 1
            self.keep_probs = 1.0
        self.num_units = options.num_units
        self.vocab_size = vocab_size
        self.input_placeholder = tf.placeholder(
            tf.int32, [self.batch_size, self.num_timesteps])
        self.target = tf.placeholder(
            tf.int32, [self.batch_size, self.num_timesteps])
        with tf.variable_scope('embed'):
            embedding = tf.get_variable(
                "embedding", [vocab_size, self.num_units])
            embedded_inputs = tf.nn.embedding_lookup(
                embedding, self.input_placeholder)
        with tf.name_scope('rnn-cell'):
            self.cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units)
            self.cell = tf.contrib.rnn.DropoutWrapper(
                self.cell, output_keep_prob=self.keep_probs)
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [self.cell] * options.num_layers)
            self.init_state = self.cell.zero_state(self.batch_size, tf.float32)
            self.rnn_output, self.final_state = tf.nn.dynamic_rnn(
                self.cell, embedded_inputs, dtype=tf.float32, initial_state=self.init_state)
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable(
                'softmax_w', [self.num_units, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            reshaped_rnn_output = tf.reshape(
                self.rnn_output, [-1, self.num_units])
            softmax_output = tf.matmul(
                reshaped_rnn_output, softmax_w) + softmax_b
        if not is_training:
            self.softmax_output = tf.nn.softmax(softmax_output)
            return
        self.softmax_output = tf.reshape(
            softmax_output, [-1, self.num_timesteps, vocab_size])

        self.loss_value = tf.contrib.seq2seq.sequence_loss(self.softmax_output,
                                                           self.target,
                                                           tf.ones([self.batch_size, self.num_timesteps]))
        opt = tf.train.AdamOptimizer(learning_rate=0.003)
        self.train_op = opt.minimize(self.loss_value)
