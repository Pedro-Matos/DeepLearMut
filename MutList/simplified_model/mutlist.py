import numpy as np
import tensorflow as tf
from preprocessing import PreProcessing
from sklearn.model_selection import train_test_split
import datetime
from random import randint

class Mutlist:
    def __init__(self):
        self.wordsList = None
        self.wordVectors = None
        self.maxSeqLength = 1  # Maximum length of sentence
        self.numDimensions = 300  # Dimensions for each word vector
        self.types = {}
        self.batchSize = 24
        self.numClasses = -1
        self.lstmUnits = 64
        self.iterations = 100000

    def main(self):
        preprocess = PreProcessing()
        self.wordsList, self.wordVectors = preprocess.load_glove()
        data, labels, types = preprocess.load_mutations()
        self.numClasses = len(types)

        # create dictionary of type and it's respective value in int
        count = 0
        for i in types:
            dic = {i: count}
            self.types.update(dic)
            count = count + 1

        train_seqs, test_seqs, train_labels, test_labels = self.normalize_data(data, labels)

        self.prepare_lstm(train_seqs, train_labels, test_seqs, test_labels)


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
        dt = np.zeros([self.batchSize, self.maxSeqLength])

        for i in range(self.batchSize):
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
        dt = np.zeros([self.batchSize, self.maxSeqLength])

        for i in range(self.batchSize):
            num = randint(0, len(test_seqs) - 1)
            if test_labels[num] == 0:
                labels.append([1, 0, 0])
            elif test_labels[num] == 1:
                labels.append([0, 1, 0])
            elif test_labels[num] == 2:
                labels.append([0, 0, 1])
            dt[i] = test_seqs[num]

        return dt, labels

    def prepare_lstm(self, train_seqs, train_labels, test_seqs, test_labels):
        # define the network
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

        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # add TensorBoard to visualize the loss and accuracy values
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


        # run the network
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for i in range(self.iterations):
            # Next batch of reviews
            nextBatch, nextBatchLabels = self.get_TrainingBatch(train_seqs, train_labels)
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            # Write summary to TensorBoard
            if i % 50 == 0:
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if i % 10000 == 0 and i != 0:
                save_path = saver.save(sess, "/Users/pmatos9/Desktop/pedrinho/tese/checkpoints/MutList/pretrained_simple_lstm.ckpt",
                                       global_step=i)
                print("saved to %s" % save_path)

        writer.close()

        print("\n\n")
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("/Users/pmatos9/Desktop/pedrinho/tese/checkpoints/MutList/"))

        iterations = 10
        for i in range(iterations):
            nextBatch, nextBatchLabels = self.get_TestBatch(test_seqs, test_labels)
            print("Accuracy for this batch:",
                  (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)


if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
