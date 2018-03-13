import numpy as np
import tensorflow as tf
from preprocessing import PreProcessing
from sklearn.model_selection import train_test_split
from random import randint
import sys
import time

class Teste:
    def __init__(self):
        self.maxSeqLength = 1  # Maximum length of sentence
        self.types = {}
        self.numClasses = -1
        # unrolled through x time steps
        self.time_steps = 1
        # hidden LSTM units
        self.num_units = 64
        # learning rate for adam
        self.learning_rate = 0.001
        # size of batch
        self.batch_size = 24
        self.iterations = 1000

    def main(self):
        preprocess = PreProcessing()
        data, labels, types = preprocess.load_mutations()
        self.numClasses = len(types)

        # create dictionary of type and it's respective value in int
        count = 0
        for i in types:
            dic = {i: count}
            self.types.update(dic)
            count = count + 1

        train_seqs, test_seqs, train_labels, test_labels = self.normalize_data(data, labels)

        # Spit out details about data
        classes = np.sort(np.unique(train_labels))
        print("\n=================================\nData details:")
        print("- Training-set:\t{}".format(len(train_seqs)))
        print("- Test-set:\t\t{}".format(len(test_seqs)))
        print("- Classes:\t\t{}".format(classes))
        print("=================================\n\n")

        self.train(train_seqs, test_seqs, train_labels, test_labels)


    def normalize_data(self, data, labels):

        # split the data to train and to test
        train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
        labels_train, labels_test = train_test_split(labels, test_size=0.1, shuffle=False)

        mut_set = list(set(data))
        vocab_size = len(mut_set)

        # Create a dictionary of the mutations and their index value in the list
        # This way it's possible to have a numeric code for each letter.
        vocab = dict(zip(mut_set, range(vocab_size)))

        # train_seqs = np.array([list(map(vocab.get, k)) for k in train_df])
        # test_seqs = np.array([list(map(vocab.get, k)) for k in test_df])
        train_seqs = np.array([vocab.get(k) for k in train_df])
        test_seqs = np.array([vocab.get(k) for k in test_df])

        # Converting categorical labels to actual numbers For two categories of
        # labels, for example, we'll have classes [0, 1].
        train_labels = np.array([self.types.get(i) for i in labels_train])
        test_labels = np.array([self.types.get(i) for i in labels_test])

        return train_seqs, test_seqs, train_labels, test_labels

    def get_TrainingBatch(self, train_seqs, train_labels):
        labels = []
        dt = np.zeros([self.batch_size, self.maxSeqLength])

        for i in range(self.batch_size):
            num = randint(0, len(train_seqs) - 1)
            if train_labels[num] == 0:
                labels.append([1, 0, 0])
            elif train_labels[num] == 1:
                labels.append([0, 1, 0])
            elif train_labels[num] == 2:
                labels.append([0, 0, 1])
            dt[i] = train_seqs[num]

        return dt, labels

    def get_TestBatch(self, test_seqs, test_labels):
        labels = []
        dt = np.zeros([self.batch_size, self.maxSeqLength])

        for i in range(self.batch_size):
            num = randint(0, len(test_seqs) - 1)
            if test_labels[num] == 0:
                labels.append([1, 0, 0])
            elif test_labels[num] == 1:
                labels.append([0, 1, 0])
            elif test_labels[num] == 2:
                labels.append([0, 0, 1])
            dt[i] = test_seqs[num]

        return dt, labels

    def train(self, train_seqs, test_seqs, train_labels, test_labels):
        start = time.time()

        # weights and biases of appropriate shape to accomplish above task
        out_weights = tf.Variable(tf.random_normal([self.num_units, self.numClasses]))
        out_bias = tf.Variable(tf.random_normal([self.numClasses]))

        # defining placeholders
        # input image placeholder
        x = tf.placeholder("float", [None, self.time_steps, self.maxSeqLength])
        # input label placeholder
        y = tf.placeholder("float", [None, self.numClasses])

        # processing the input tensor from [batch_size,n_steps,maxSeqLength] to "time_steps"
        # number of [batch_size,n_input] tensors
        input = tf.unstack(x, self.time_steps, 1)

        # defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, input, dtype="float32")

        # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes]
        # by out_weight multiplication
        prediction = tf.matmul(outputs[-1], out_weights) + out_bias

        # loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # words_vector_model evaluation
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        file = open("epoch_testModel.txt", "w")

        # initialize variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            iter = 1
            while iter < self.iterations:
                # Next batch of reviews
                next_batch, next_batch_labels = self.get_TrainingBatch(train_seqs, train_labels)
                batch_x = next_batch.reshape((self.batch_size, self.time_steps, self.maxSeqLength))
                sess.run(opt, feed_dict={x: batch_x, y: next_batch_labels})

                if iter % 100 == 0 and iter != 0:
                    print("\n\n")
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: next_batch_labels})

                    file.write("For iteration " + str(iter))
                    file.write("\n")
                    file.write("Accuracy " + str(acc))
                    file.write("\n")
                    file.write("__________________\n")
                    file.flush()
                    file.write("\n\n")

                iter = iter + 1


            iterations = 10
            for i in range(iterations):
                # calculating test accuracy
                nextBatch, nextBatchLabels = self.get_TestBatch(test_seqs, test_labels)
                batch_x = nextBatch.reshape((self.batch_size, self.time_steps, self.maxSeqLength))
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: nextBatchLabels})
                print("Testing Accuracy:", acc)
                file.write("Testing Accuracy: " + str(acc))
                file.write("\n")
                file.flush()

            file.close()

            end = time.time()
            print("Time: ")
            print(end - start)


if __name__ == "__main__":
    mutlist = Teste()
    mutlist.main()
