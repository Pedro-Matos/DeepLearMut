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

        self.k = 5  # value for k fold cross validation

        self.numDimensions = 200  # Dimensions for each word vector
        self.batchSize = 14
        self.lstmUnits = 64
        self.iterations = 5
        self.epochs = 1

    def split_data(self, data, labels):
        # split the data to train and to test
        train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
        labels_train, labels_test = train_test_split(labels, test_size=0.1, shuffle=False)

        return train_df, test_df, labels_train, labels_test

    def create_matrix_train(self, train):
        num_train = len(train)

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

        # print(ids_train.shape)

        return ids_train

    def create_matrix_teste(self, test):

        num_test = len(test)

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

        # print(ids_test.shape)

        return ids_test

    def get_train_batch(self, data, labels, ids, idx):
        l = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])

        num = idx * self.batchSize

        for i in range(self.batchSize):

            indice = num + i
            lab_list = labels[indice]
            num_list = randint(0, len(lab_list) - 1)
            label_class = self.types.get(lab_list[num_list])

            # arr[i] = ids_train[num - 1:num]   # ValueError: could not broadcast input array from shape (0,429) into shape (429)
            arr[i] = ids[indice]

            if label_class == 0:
                l.append([1, 0, 0, 0])
            elif label_class == 1:
                l.append([0, 1, 0, 0])
            elif label_class == 2:
                l.append([0, 0, 1, 0])
            elif label_class == 3:
                l.append([0, 0, 0, 1])

        return arr, l

    def train_model(self, train_bins, label_bins, ids_train):
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

        for epoch in range(self.epochs):
            acc = 0
            # we need to try all combinations with 4 as training and 1 as validation
            for idx in range(len(train_bins)):

                # 5 iterations need to get all the dataset from bin
                validation = train_bins[idx]
                validation_labels = label_bins[idx]
                validation_ids = ids_train[idx]

                tmp_train = [x for i,x in enumerate(train_bins) if i !=idx]
                tmp_labels = [x for i, x in enumerate(label_bins) if i != idx]
                tmp_ids = [x for i, x in enumerate(ids_train) if i != idx]

                for j in range(len(tmp_train)):
                    for i in range(self.iterations):
                        nextBatch, nextBatchLabels = self.get_train_batch(tmp_train[j], tmp_labels[j], tmp_ids[j], i)
                        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

                # calculate accuracy with the validation data
                acc_bin = 0
                for i in range(self.iterations):
                    nextBatch, nextBatchLabels = self.get_train_batch(validation, validation_labels, validation_ids, i)
                    tmp_acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}) * 100
                    acc_bin = acc_bin + tmp_acc
                acc_bin = acc_bin / 5
                acc = acc + acc_bin
            f_acc = acc / len(train_bins)
            print("Accuracy for epoch %s is %s" % (str(epoch), str(f_acc)))




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

        # remove last element to make it even number
        train_df = train_df[:-1]
        labels_train = labels_train[:-1]
        div = int(len(train_df) / self.k)

        # get K = 5 batches
        train_1, train_2, train_3, train_4, train_5 = [train_df[i:i + div] for i in range(0, len(train_df), div)]
        labels_1, labels_2, labels_3, labels_4, labels_5 = [labels_train[i:i + div] for i in range(0, len(labels_train), div)]

        ids_test = self.create_matrix_teste(test_df)
        ids_train1 = self.create_matrix_train(train_1)
        ids_train2 = self.create_matrix_train(train_2)
        ids_train3 = self.create_matrix_train(train_3)
        ids_train4 = self.create_matrix_train(train_4)
        ids_train5 = self.create_matrix_train(train_5)

        train_bins = [train_1, train_2, train_3, train_4, train_5]
        label_bins = [labels_1, labels_2, labels_3, labels_4, labels_5]
        ids_train = [ids_train1, ids_train2, ids_train3, ids_train4, ids_train5]

        self.train_model(train_bins, label_bins, ids_train)

if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
