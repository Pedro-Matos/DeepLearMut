from preprocess import PreProcess
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import datetime
from random import randint

class Mutlist:
    def __init__(self):
        self.wordsList = None
        self.wordVectors = None

        self.numClasses = -1
        self.types = {}
        self.maxSeqLength = -1  # Maximum length of sentence

        self.numDimensions = 200  # Dimensions for each word vector
        self.batchSize = 24
        self.lstmUnits = 64
        self.iterations = 100000

    def create_matrix(self, train, test):
        num_train = len(train)
        num_test = len(test)

        ids_train = np.zeros((num_train, self.maxSeqLength), dtype='int32')
        file_counter = 0

        for index in train:
            token_counter = 0
            for token in index:
                try:
                    ids_train[file_counter][token_counter] = self.wordsList.index(token)
                except ValueError:
                    ids_train[file_counter][token_counter] = 499999  # full of zeros
                token_counter = token_counter + 1

                if token_counter >= self.maxSeqLength:
                    break

            file_counter = file_counter + 1

        ids_test = np.zeros((num_test, self.maxSeqLength), dtype='int32')
        file_counter = 0

        for index in test:
            token_counter = 0
            for token in index:
                try:
                    ids_test[file_counter][token_counter] = self.wordsList.index(token)
                except ValueError:
                    ids_test[file_counter][token_counter] = 499999  # full of zeros
                token_counter = token_counter + 1

                if token_counter >= self.maxSeqLength:
                    break

            file_counter = file_counter + 1

        print(ids_train.shape)
        print(ids_test.shape)

        return ids_train, ids_test

    def split_data(self, data, labels):
        # split the data to train and to test
        train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
        labels_train, labels_test = train_test_split(labels, test_size=0.1, shuffle=False)

        return train_df, test_df, labels_train, labels_test

    def get_train_batch(self, ids_train, labels):
        l = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])

        for i in range(self.batchSize):
            if i < (self.batchSize / 3):
                num = randint(0, 93)
                lab_list = labels[num]
                num_list = randint(0, len(lab_list)-1)
                label_class = self.types.get(lab_list[num_list])

            elif i < (self.batchSize / 3) * 2:
                num = randint(94, 187)
                lab_list = labels[num]
                num_list = randint(0, len(lab_list)-1)
                label_class = self.types.get(lab_list[num_list])

            elif i < (self.batchSize / 3) * 3:
                num = randint(188, 281)
                lab_list = labels[num]
                num_list = randint(0, len(lab_list)-1)
                label_class = self.types.get(lab_list[num_list])

            # arr[i] = ids_train[num - 1:num]   # ValueError: could not broadcast input array from shape (0,429) into shape (429)
            arr[i] = ids_train[num]

            if label_class == 0:
                l.append([1, 0, 0])
            elif label_class == 1:
                l.append([0, 1, 0])
            elif label_class == 2:
                l.append([0, 0, 1])

        return arr, l

    def train_model(self, ids_train, labels_train, ids_test, labels_test):
        tf.reset_default_graph()

        labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])

        data = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.wordVectors, input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        # The correct prediction formulation works by looking at the index of the maximum value of the output values,
        #  and then seeing whether it matches with the training labels.
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # create the Session to run the graph
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())



        for i in range(self.iterations):
            # Next Batch 
            nextBatch, nextBatchLabels = self.get_train_batch(ids_train, labels_train)
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            print(i)
            if (i % 1000 == 0 and i != 0):
                print("Already done %s" % i)

            #Save the network every 10,000 training iterations
            if (i % 10000 == 0 and i != 0):
                save_path = saver.save(sess, "/Users/pmatos9/Desktop/pedrinho/tese/checkpoints/MutList/words/pretrained_lstm.ckpt",
                                       global_step=i)
                print("saved to %s" % save_path)

        # sess = tf.InteractiveSession()
        # saver = tf.train.Saver()
        # saver.restore(sess, tf.train.latest_checkpoint('models'))
        #
        # iterations = 10
        # for i in range(iterations):
        #     nextBatch, nextBatchLabels = getTestBatch()
        #     print("Accuracy for this batch:",
        #           (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

    def main(self):
        preprocess = PreProcess()
        self.wordsList, self.wordVectors = preprocess.load_word2vec()
        data, list_id, labels, types, max_length = preprocess.load_mutations()
        self.maxSeqLength = max_length + 1

        # create dictionary of type and it's respective value in int
        count = 0
        for i in types:
            dic = {i: count}
            self.types.update(dic)
            count = count + 1

        self.numClasses = len(self.types)

        train_df, test_df, labels_train, labels_test = self.split_data(data, labels)
        ids_train, ids_test = self.create_matrix(train_df, test_df)

        # Spit out details about data
        print("\n=================================\nData details:")
        print("- Training-set:\t{}".format(len(train_df)))
        print("- Test-set:\t\t{}".format(len(test_df)))
        print("- Classes:\t\t{}".format(self.types))
        print("=================================\n\n")

        self.train_model(ids_train, labels_train, ids_test, labels_test)

if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
