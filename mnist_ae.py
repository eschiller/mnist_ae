import tensorflow as tf
import tensorflow.examples.tutorials.mnist


class mnist_ae:
    def __init__(self, hlayers=196, learn_rate=0.1, optimizer=tf.train.GradientDescentOptimizer):
        self.mnist_data = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.x = tf.placeholder(tf.float32, shape=[None, 784])

        self.wz = tf.Variable(tf.random_normal([784, hlayers]), dtype=tf.float32)
        self.bz = tf.Variable(tf.zeros([hlayers], dtype=tf.float32))

        self.z = tf.tanh(tf.matmul(self.x, self.wz))# + self.bz

        self.wy = tf.Variable(tf.random_normal([hlayers, 784], dtype=tf.float32))
        self.by = tf.Variable(tf.zeros([784], dtype=tf.float32))

        self.y = tf.sigmoid(tf.matmul(self.z, self.wy))# + self.by

        self.cost = tf.reduce_mean(tf.square(self.x - self.y))
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(x * tf.log(y), reduction_indices=[1]))

        self.train_step = optimizer(learn_rate).minimize(self.cost)

        #PREP
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    #TRAIN
    def train(self, reps):
        for i in range(reps):
            batch_xs, batch_ys = self.mnist_data.train.next_batch(50)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs})

    def reconstruct_num(self, num):
        reconstructed_num = self.sess.run(self.y, feed_dict={self.x: num})
        return reconstructed_num

    def get_num_and_recon(self):
        image_orig = self.mnist_data.test.next_batch(1)[0]
        image = self.reconstruct_num(image_orig)
        return image_orig, image

    def get_weights_z(self):
        return self.sess.run(self.wz)

    def get_weights_y(self):
        return self.sess.run(self.wy)

    def get_x_and_y(self):
        return self.sess.run((self.x, self.y))

    def get_next_z(self):
        image = self.mnist_data.test.next_batch(1)[0]
        return self.sess.run(self.z, feed_dict={self.x: image})

    def get_bias(self):
        return self.sess.run((self.by, self.bz))

    def get_all_graph_values(self):
        image = self.mnist_data.test.next_batch(1)[0]
        gz, gy, gwz, gwy, gbz, gby = self.sess.run((self.z, self.y, self.wz, self.wy, self.bz, self.by), feed_dict={self.x: image})
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
