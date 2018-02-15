import tensorflow as tf


class Model:
    def __init__(self):
        self.num_hidden = 24
        self.batch_size = 1000
        self.epoch = 5000

    def create_model(self,train_input, train_output, test_input, test_output):
        # we will define two variables which will hold the input data and the target data.
        data = tf.placeholder(tf.float32, [None, 20, 1])
        target = tf.placeholder(tf.float32, [None, 21])

        # create the LSTM model

        cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden,state_is_tuple=True)


        '''
        The first phase is building the computation graph where you define all the calculations and functions 
        that you will execute during runtime. The second phase is the execution phase where a Tensorflow session
         is created and the graph that was defined earlier is executed with the data we supply.
        '''

        val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([self.num_hidden, int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

        cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(cross_entropy)

        # Calculating the error on test data
        '''
        This error is a count of how many sequences in the test dataset were classified incorrectly.
        This gives us an idea of the correctness of the model on the test dataset.
        '''

        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))






        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)


        no_of_batches = int(len(train_input) / self.batch_size)

        for i in range(self.epoch):
            ptr = 0
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr + self.batch_size], train_output[ptr:ptr + self.batch_size]
                ptr += self.batch_size
                sess.run(minimize, {data: inp, target: out})
            print( "Epoch - ", str(i))
        incorrect = sess.run(error, {data: test_input, target: test_output})
        print(sess.run(prediction, {data: [[[1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0]]]}))
        print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
        sess.close()
