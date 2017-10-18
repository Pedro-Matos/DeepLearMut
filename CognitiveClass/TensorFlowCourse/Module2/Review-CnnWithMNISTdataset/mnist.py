import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


## 1st PART:  classify MNIST using a simple model; USING INTERACTIVE SESSION


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])   # represents the "space" allocated input or the images.
y_ = tf.placeholder(tf.float32, shape=[None, 10])    # represents the final output or the labels.

# weight tensor
W = tf.Variable(tf.zeros([784,10], tf.float32))
# bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))
sess.run(tf.global_variables_initializer())

# mathematical operation to add weights and biases to the inputs
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Load 50 training examples for each training iteration
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc))
sess.close() #finish the session