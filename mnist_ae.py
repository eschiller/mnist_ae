import tensorflow as tf
import numpy as np

import tensorflow.examples.tutorials.mnist
import mnist_imaging

mnist_data = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])

wz = tf.Variable(tf.random_normal([784, 200]), dtype=tf.float32)
bz = tf.Variable(tf.zeros([200], dtype=tf.float32))

mult = tf.matmul(x, wz)
add = mult# + bz

#z = tf.nn.tanh(add)
z = (add)


#wy = tf.Variable(tf.random_normal([20, 784], dtype=tf.float32))
wy = tf.transpose(wz)
by = tf.Variable(tf.zeros([784], dtype=tf.float32))

mult2 = tf.matmul(z, wy)
add2 = mult2# + by


y = (tf.sigmoid(add2))

cost = tf.reduce_mean(tf.square(x - y))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(x * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

#PREP
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#TRAIN
def train(reps):
    for i in range(reps):
        batch_xs, batch_ys = mnist_data.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch_xs})


def reconstruct_num(num):
    reconstructed_num = sess.run(y, feed_dict={x: num})
    return reconstructed_num


def get_num_and_recon():
    image_orig = mnist_data.test.next_batch(1)[0]
    image = reconstruct_num(image_orig)
    return image_orig, image


def get_weights_z():
    return sess.run(wz)


def get_weights_y():
    return sess.run(wy)


def get_x_and_y():
    return sess.run((x, y))


def get_next_z():
    image = mnist_data.test.next_batch(1)[0]
    return sess.run(z, feed_dict={x: image})


def get_bias():
    return sess.run((by, bz))


def get_all_graph_values():
    image = mnist_data.test.next_batch(1)[0]
    gz, gy, gwz, gwy, gbz, gby = sess.run(( z, y, wz, wy, bz, by), feed_dict={x: image})
    print("\nX VALUES:")
    print(image)
    print("\nZ VALUES:")
    print(gz)
    print("\nY VALUES:")
    print(gy)
    print("\nZ BIAS:")
    print(gbz)
    print("\nY BIAS:")
    print(gby)
    print("\nWEIGHTS TO Z:")
    print(gwz)
    print("\nWEIGHTS TO Y:")
    print(gwy)
