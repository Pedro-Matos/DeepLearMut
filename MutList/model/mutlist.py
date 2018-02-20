import numpy as np
import tensorflow as tf
from preprocess import PreProcess
import matplotlib.pyplot as plt
import math
import datetime

class MutList:
    def __init__(self):
        self.model = "lstm"
        self.wordsList = None
        self.wordVectors = None
        self.dic_text = None
        self.list_id = None
        self.dic_results = None
        self.list_types = None
        self.maxSeqLength = -1  # Maximum length of sentence
        self.numDimensions = 300  # Dimensions for each word vector

        self.batchSize = 24
        self.numClasses = -1
        self.lstmUnits = 64
        self.iterations = 100000

    def main(self):
        print("Starting")
        prep = PreProcess()
        self.wordsList, self.wordVectors = prep.load_glove()
        self.dic_text, self.list_id, self.dic_results, self.list_types = prep.load_mutations()
        self.maxSeqLength = self.round_int(int(prep.average_words))
        self.numClasses = len(self.list_types)

        ids = self.read_file_by_file()




    def round_int(self, x):
        return 10 * ((x + 5) // 10)

    def read_file_by_file(self):
        ids = np.zeros((len(self.list_id), self.maxSeqLength), dtype='int32')


        for i in self.list_id:
            file = np.zeros(self.maxSeqLength, dtype='int32')
            indexCounter = 0
            fileCounter = 0

            for word in self.dic_text[i]:
                try:
                    file[indexCounter] = self.wordsList.index(word)
                except ValueError:
                    file[indexCounter] = 399999  # Vector for unknown words
                indexCounter = indexCounter + 1
                if indexCounter >= self.maxSeqLength:
                    break

            # print(file.shape)
            fileCounter = fileCounter + 1
            ids[fileCounter] = file

        print(ids.shapes)
        return ids

    def prepare_lstm(self):
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
            nextBatch, nextBatchLabels = self.getTrainBatch()
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            # Write summary to TensorBoard
            if i % 50 == 0:
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if i % 10000 == 0 and i != 0:
                save_path = saver.save(sess, "/Users/pmatos9/Desktop/pedrinho/tese/checkpoints/MutList/pretrained_lstm.ckpt",
                                       global_step=i)
                print("saved to %s" % save_path)

        writer.close()

    def getTrainBatch(self):
        labels = []








if __name__ == "__main__":
    mutlist = MutList()
    mutlist.main()
