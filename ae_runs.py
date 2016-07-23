import mnist_ae
import mnist_imaging
import tensorflow as tf


def ae_run(train_steps, hlayers, weight_rate, bias_rate, optimizer):
    ae = mnist_ae.mnist_ae(hlayers=hlayers, learn_rate=weight_rate, b_rate=bias_rate, optimizer=optimizer)
    ae.train(train_steps)
    ae.get_all_graph_values()
    im1, im2 = ae.get_num_and_recon()
    mnist_imaging.save_image(im1, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_01orig.bmp")
    mnist_imaging.save_image(im2, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_01recon.bmp")
    im1, im2 = ae.get_num_and_recon()
    mnist_imaging.save_image(im1, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_02orig.bmp")
    mnist_imaging.save_image(im2, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_02recon.bmp")
    im1, im2 = ae.get_num_and_recon()
    mnist_imaging.save_image(im1, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_03orig.bmp")
    mnist_imaging.save_image(im2, "TS" + str(train_steps) + "_HL" + str(hlayers) + "_LR" + str(weight_rate) + "_BM" + str(bias_rate) + "_03recon.bmp")


ae = ae_run(400000, 196, 0.5, .001, tf.train.GradientDescentOptimizer)

