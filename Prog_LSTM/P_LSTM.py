'''Progressive LSTM
Progressive Neural Network is proposed on
    Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016).
    Progressive neural networks. arXiv preprint arXiv:1606.04671.

Progressive NN is implemented simply on
    https://github.com/synpon/prog_nn

Implement Progressive NN + LSTM => P_LSTM
'''
import tensorflow as tf
from tensorflow.contrib import rnn

def lstm_cell(num_units, name):
    cell = rnn.LSTMCell(num_units, state_is_tuple=True, activation=tf.tanh, name=name)
#    print(cell, num_units)
    return cell

def weight_variable(shape, stddev=0.1, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape, init_bias=0.1, initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

class load_graph:
    def __init__(self, X, X2, X3, Y, Y2, Y3,
                 seq_length, seq_length_3, hidden_size_1, hidden_size_2, hidden_size_3,
                 learning_rate, learning_rate2, learning_rate3, train_summary):

        self.train_summary = train_summary
        self.col_1 = InitialColumn(input_x=X,
                              floor_num=3,
                              hidden_size=hidden_size_1,
                              case=1
                              )
        # train_summary_col_1.append(col_1.train_summary)
        var_list_1 = tf.trainable_variables(scope=None)  # 현재까지의 trainable Variable List

        self.col_2 = ExtensibleColum(input_x=X2,
                                    n=2,
                                    floor_num=3,
                                    hidden_size=hidden_size_2,
                                    seq_length=seq_length,
                                    prev_columns=[self.col_1],
                                    )
        var_list_2 = tf.trainable_variables(scope=None)
        var_only_in_2 = list(set(var_list_2) - set(var_list_1)) # var list only in self.col_2

        self.col_3 = ExtensibleColum(input_x=X3,
                                       n=3,
                                       floor_num=3,
                                       hidden_size=hidden_size_3,
                                       seq_length=seq_length_3,
                                       prev_columns=[self.col_1, self.col_2],
                                       )
        var_list_3 = tf.trainable_variables(scope=None)
        var_only_in_3 = list(set(var_list_3) - set(var_only_in_2)) # var list only in self.col_3

        with tf.name_scope("Cost") as cost_scope:

            # for L2 Normalization
            # l2_1 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in var_list_1)
            self.l2_2 = 0.00001 * sum(tf.nn.l2_loss(tf_var) for tf_var in var_only_in_2)
            self.l2_3 = 0.00001 * sum(tf.nn.l2_loss(tf_var) for tf_var in var_only_in_3)

            self.cost_1 = tf.reduce_mean(tf.square(self.col_1.h[-1] - Y)) / 2
            self.cost_2 = tf.reduce_mean(tf.square(self.col_2.h[-1] - Y2)) / 2 + self.l2_2
            self.cost_3 = tf.reduce_mean(tf.square(self.col_3.h[-1] - Y3)) / 2 + self.l2_3

            self.cost_scalar = tf.summary.scalar('cost_1', self.cost_1)
            self.cost_scalar_2 = tf.summary.scalar('cost_2', self.cost_2)
            self.cost_scalar_3 = tf.summary.scalar('cost_3', self.cost_3)
            self.train_summary.append(self.cost_scalar)
            self.train_summary.append(self.cost_scalar_2)
            self.train_summary.append(self.cost_scalar_3)

        with tf.name_scope("train_1") as train_scope_1:
            optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_1 = optimizer_1.minimize(self.cost_1, var_list=var_list_1)

        with tf.name_scope("train_2") as train_scope_2:
            optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
            self.train_2 = optimizer_2.minimize(self.cost_2, var_list=var_only_in_2)

        with tf.name_scope("train_3") as train_scope_3:
            optimizer_3 = tf.train.AdamOptimizer(learning_rate=learning_rate3)
            self.train_3 = optimizer_3.minimize(self.cost_3, var_list=var_only_in_3)

class InitialColumn(object):
    def __init__(self, input_x, floor_num, hidden_size, case):
        self.h = [input_x]
        self.hidden_size = hidden_size
        # self.session = session
        self.train_summary = []

        if case is 1:
            with tf.variable_scope("column_1"):
                for idx in range(1, floor_num + 1):
                    with tf.name_scope("LSTM_{}_{}".format(1, idx)) as lstm_scope:
                        cell = lstm_cell(self.hidden_size, "LSTM_Cell_{}_{}".format(1, idx))
                        lstm_outputs, _states = tf.nn.dynamic_rnn(cell, input_x, dtype=tf.float32)
                        self.h.append(lstm_outputs)

                        input_x = lstm_outputs

                with tf.name_scope("Fully_Connected_1") as fully_connected_scope:
                    fully_input = lstm_outputs[:, -1]
                    Y_predict = tf.contrib.layers.fully_connected(fully_input, 1, activation_fn=tf.nn.leaky_relu,
                                                                  scope=fully_connected_scope)
                    self.h.append(Y_predict)
                    for var in tf.trainable_variables(fully_connected_scope):
                        fully_connected_summary = tf.summary.histogram(var.name, var)
                        self.train_summary.append(fully_connected_summary)

class ExtensibleColum(object):
    def __init__(self, input_x, n, floor_num, hidden_size, seq_length, prev_columns):
        self.h = [input_x]
        self.hidden_size = hidden_size
        self.prev_columns = prev_columns
        self.width = len(prev_columns)
        # self.session = session
        self.train_summary = []
        self.U = [] # lateral weight matrix
        for _ in range(floor_num):
            self.U.append( [[]] * self.width)
        with tf.variable_scope("column_{}".format(n)):
            for idx in range(1, floor_num + 1):
                with tf.name_scope("LSTM_{}_{}".format(n, idx)) as lstm_scope:
                    cell = lstm_cell(self.hidden_size, "LSTM_Cell_{}_{}".format(n, idx))
                    lstm_outputs, _states = tf.nn.dynamic_rnn(cell, input_x, dtype=tf.float32)
                    if idx == 1:
                        preactivation = lstm_outputs # (?, 100, 10)
                        self.h.append(lstm_outputs)
                    else:
                        for idx2 in range(self.width): # iterative on previous columns
                            prev_columns_trans = tf.transpose(prev_columns[idx2].h[idx-1], [1, 0, 2]) # ( 100, ? ,h)
                            self.U[idx-1][idx2] = tf.cast(weight_variable([seq_length, self.prev_columns[0].hidden_size
                                                                              , self.hidden_size]), tf.float32)
                            a = []
                            for k in range(seq_length):
                                a.append(tf.matmul(prev_columns_trans[k], self.U[idx-1][idx2][k])) # matmul((?, h), (h, h))
                            a = tf.reshape(a, [-1, seq_length, self.hidden_size])
                            lstm_outputs = tf.add(lstm_outputs, a)
                            self.h.append(tf.nn.relu(lstm_outputs)) # relu activation 함수 층 중간에 존재함

                    input_x = lstm_outputs

        with tf.name_scope("Fully_Connected_2") as fully_connected_scope:
            fully_input = lstm_outputs[:, -1]
            Y_predict = tf.contrib.layers.fully_connected(fully_input, 1, activation_fn=tf.nn.leaky_relu,
                                                          scope=fully_connected_scope)
            self.h.append(Y_predict)
            for var in tf.trainable_variables(fully_connected_scope):
                fully_connected_summary = tf.summary.histogram(var.name, var)
                self.train_summary.append(fully_connected_summary)

if __name__ == "__main__":
    Y = tf.placeholder(tf.float32, [None, 1], name='y_input')
    Y2 = tf.placeholder(tf.float32, [None, 1], name='predict_cycle_2')
    Y3 = tf.placeholder(tf.float32, [None, 1], name='predict_cycle_3')

    X = tf.placeholder(tf.float32, [None, 20, 10], name='x_input_1')
    X2 = tf.placeholder(tf.float32, [None, 20, 1], name='x_input_2')
    X3 = tf.placeholder(tf.float32, [None, 20, 1], name='x_input_3')
    hidden_size_1 = 10
    hidden_size_2 = 10
    hidden_size_3 = 10
    cell_num = 3
    train_summary = []

    with tf.Session() as sess:
        model = load_graph(X=X, X2=X2, Y=Y, Y2=Y2, X3=X3, Y3=Y3,
                           seq_length=20,
                           seq_length_3=20,
                           hidden_size_1=hidden_size_1,
                           hidden_size_2=hidden_size_2,
                           hidden_size_3=hidden_size_3,
                           learning_rate=0.0001,
                           learning_rate2=0.0001,
                           learning_rate3=0.0001,
                           train_summary=train_summary,
                           )

        c1_result = model.col_1.h[-1]
        c2_result = model.col_2.h[-1]
        c3_result = model.col_3.h[-1]