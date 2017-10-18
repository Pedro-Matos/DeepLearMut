import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Architecture of our network is:
(Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
(Convolutional layer 1) -> [batch_size, 28, 28, 32]
(ReLU 1) -> [?, 28, 28, 32]
(Max pooling 1) -> [?, 14, 14, 32]
(Convolutional layer 2) -> [?, 14, 14, 64]
(ReLU 2) -> [?, 14, 14, 64]
(Max pooling 2) -> [?, 7, 7, 64]
[fully connected layer 3] -> [1x1024]
[ReLU 3] -> [1x1024]
[Drop out] -> [1x1024]
[fully connected layer 4] -> [1x10]
'''

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# INITIAL PARAMETERS
width = 28  # width of the image in pixels
height = 28 # height of the image in pixels
flat = width * height   # number of pixels in one image
class_output = 10   # number of possible classifications for the problem

x = tf.placeholder(tf.float32, shape=[None, flat])
y = tf.placeholder(tf.float32, shape=[None, class_output])

## The input image is a 28 pixels by 28 pixels, 1 channel (grayscale).
# In this case, the first dimension is the batch number of the image,
# and can be of any size (so we set it to -1). The second and third dimensions are width and hight,
# and the last one is the image channels.
x_image = tf.reshape(x, [-1,28,28,1])

###             FIRST CONVOLUTIONAL LAYER                ###
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))   #filter?
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1) # wherever a negative number occurs,we swap it out for a 0. (ReLU1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2 (Max pooling 1)


###             SECOND CONVOLUTIONAL LAYER                ###
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) # need 64 biases for 64 outputs
convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2) # ReLU2
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   #max_pool_2x2


###             FULLY CONNECTED LAYER                ###



