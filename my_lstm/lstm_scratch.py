import tensorflow as tf
import numpy as np
class _my_lstm_cell_origin():
    def __init__(self, feature_size, hidden_size, idx):
        self._hidden_size = hidden_size
        self._feature_size = int(feature_size)

        with tf.variable_scope("lstm_cell_no_{}".format(idx)):
            '''input gate weights
            f : feature_size
            h : hidden_size
            Wi = ( f + h , h )
            '''
            self.Wxi = tf.get_variable('Wxi', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whi = tf.get_variable('Whi', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wci = tf.get_variable('Wci', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wi = tf.concat([self.Wxi, self.Whi], axis=0)

            '''forget gate weights
            Wf = ( f + h  , h )
            '''
            self.Wxf = tf.get_variable('Wxf', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whf = tf.get_variable('Whf', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wcf = tf.get_variable('Wcf', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wf = tf.concat([self.Wxf, self.Whf], axis=0)

            '''cell update weights
            Wc = ( f + h , h )
            '''
            self.Wxc = tf.get_variable('Wxc', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Whc = tf.get_variable('Whc', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wc = tf.concat([self.Wxc, self.Whc], axis=0)

            '''output gate weights
            Wc = ( f + h  , h )
            '''
            self.Wxo = tf.get_variable('Wxo', shape=(self._feature_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Who = tf.get_variable('Who', shape=(self._hidden_size, self._hidden_size),
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.Wco = tf.get_variable('Wco', shape=(self._hidden_size, self._hidden_size),
            #                       initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.Wo = tf.concat([self.Wxo, self.Who], axis=0)

            '''bias term for all gates'''
            self.bi = tf.get_variable('bi', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bf = tf.get_variable('bf', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bc = tf.get_variable('bc', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.bo = tf.get_variable('bo', shape=(1, self._hidden_size),
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def in_fo_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: gate output
        '''
        ig_result = tf.sigmoid(tf.matmul(x, self.Wi) + self.bi)
        fg_result = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)

        return ig_result, fg_result

    def cell_update(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t-1)
        :return: cell state updating term
        '''
        return tf.tanh(tf.matmul(x, self.Wc) + self.bc)

    def ou_gate(self, x):
        '''
        :param x: input(t) + hidden(t-1) + cell_state(t)
        :return: output gate output
        '''
        return tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)

    def step(self, prev, input_t):
        h_state, c_state = tf.unstack(prev)
        concat_x = tf.concat([input_t, h_state], axis=1)
        ig, fg = _my_lstm_cell.in_fo_gate(self, concat_x)
        cu = _my_lstm_cell.cell_update(self, concat_x)

        cell_state_new = tf.multiply(fg, c_state) + tf.multiply(ig, cu)
        og = _my_lstm_cell.ou_gate(self, concat_x)

        h_state_new = tf.multiply(og, tf.tanh(cell_state_new))
        return tf.stack([h_state_new, cell_state_new])

    def __call__(self, rnn_input): # no keep_prob_list
        # rnn input : [?, seq_length, hidden_size]
        zeros_dims = tf.stack([2, tf.shape(rnn_input)[0], self._hidden_size])
        init_state = tf.fill(zeros_dims, 0.0)
        self._seq_length = rnn_input.shape[1]

        # scan은 반드시 init_state의 dimension 과 fn(step)의 리턴값의 dimension 이 동일해야한다.
        # scan의 initializer term은 step의 첫번째 인수에 해당한다.
        # transpose to (seq_length, ?, feature_size)
        h_and_c = tf.scan(self.step, tf.transpose(rnn_input, [1, 0, 2]), initializer=init_state)

        # need to transpose (seq_length, 2, ?, feature_size) -> (2, ?, seq_length, feature_size)
        hidden_states, cell_states = tf.unstack(tf.transpose(h_and_c, [1, 2, 0, 3]))

        return hidden_states, cell_states

if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 100, 3], name='x_input')

    lstm_cell = _my_lstm_cell(3, 10)
    hs, cs = lstm_cell(X)