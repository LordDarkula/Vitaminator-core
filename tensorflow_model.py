import tensorflow as tf
import numpy as np
from process_images import randomly_assign_train_test
from bottleneck_keras import save_images_to_arrays

x = tf.placeholder(tf.float32, shape=[None, 130 * 130])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def build_model(image_size):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([33 * 33 * 64, 1024])
    b_fc1 = bias_variable([1024])
    print h_pool2.shape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 33 * 33 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy


if __name__ == '__main__':
    # randomly_assign_train_test('images')
    train_X, train_y, test_X, test_y = save_images_to_arrays()
    train_X = np.reshape(train_X, (-1, 130, 130))
    test_X = np.reshape(test_X, (-1, 130, 130))

    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_y = test_y.astype(np.float32)

    train_X = np.reshape(train_X, [-1, 130 * 130])
    test_X = np.reshape(test_X, [-1, 130 * 130])

    train_step, accuracy = build_model(130)

    batch_size = 20
    n_batches = (len(train_X) / batch_size) - 2

    with tf.Session() as sess:
        print("Session starting")

        sess.run(tf.global_variables_initializer())
        for i in range(n_batches):
            print("On iteration number {}".format(i))
            batch = (train_X[batch_size * i:batch_size * (i + 1)], train_y[batch_size * i:batch_size * (i + 1)])
            train_accuracy = accuracy.eval(
                    feed_dict={
                        x: batch[0],
                        y_: batch[1],
                        keep_prob: 0.5}
                )

            print("step "+str(i) + ", training accuracy " + str(train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: test_X, y_: test_y, keep_prob: 1.0}))
