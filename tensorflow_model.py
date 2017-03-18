import tensorflow as tf
from process_images import randomly_assign_train_test
from bottleneck_keras import save_images_to_arrays


x = tf.placeholder(tf.float32, shape=[None, 130 * 130], name='x_placeholder')
y_ = tf.placeholder(tf.float32, shape=[None, 2], name='y_placeholder')
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='W')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='B')


def conv_layer(X, W, b, name='conv'):
    with tf.name_scope(name):
        convolution = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(convolution + b)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)

        return tf.nn.max_pool(activation, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(X, W, b, name='fc'):
    with tf.name_scope(name):
        return tf.nn.relu(tf.matmul(X, W) + b)


def build_model(image_size):
    x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    model = conv_layer(x_image, W_conv1, b_conv1, name='conv1')

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    model = conv_layer(model, W_conv2, b_conv2, name='conv2')

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    model = conv_layer(model,W_conv3,b_conv3,name='conv3')

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])


    model = conv_layer(model,W_conv4,b_conv4,name='conv4')

    W_fc1 = weight_variable([9 * 9 * 256, 1024])
    b_fc1 = bias_variable([1024])
    print model.shape

    model = tf.reshape(model, [-1, 9 * 9 * 256])
    model = fc_layer(model, W_fc1, b_fc1)
    print model.shape

    W_fc1_5 = weight_variable([1024, 1024])
    b_fc1_5 = bias_variable([1024])
    model = fc_layer(model, W_fc1_5, b_fc1_5)

    model = tf.nn.dropout(model, keep_prob)

    W_fc1_6 = weight_variable([1024, 1024])
    b_fc1_6 = bias_variable([1024])
    model = fc_layer(model, W_fc1_6, b_fc1_6)
    
    model = tf.nn.dropout(model, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(model, W_fc2) + b_fc2

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return optimizer, cross_entropy, accuracy, prediction , y_conv


def next_batch(batch_size, i, data):
    return data[batch_size * i: batch_size * (i + 1)]


def run_tensorflow_model():
    train_X, train_y, test_X, test_y = save_images_to_arrays()

    optimizer, cost, accuracy, prediction, y_conv = build_model(130)

    batch_size = 20
    n_batches = (len(train_X) / batch_size) - 2
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Session starting")

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/vitaminator/official9')

        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(500):
            epoch_loss = 0
            avg_cost = 0.0
            for i in range(n_batches):
                print('batch number: {}'.format(i))
                batch_x, batch_y = next_batch(
                    batch_size, i, train_X), next_batch(batch_size, i, train_y)

                sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.4})
                if i % 5 == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.4})
                    writer.add_summary(s, i)

            print('Epoch {} completed out of {}'.format(
                epoch + 1, 500, epoch_loss))

        print('Accuracy:', accuracy.eval({x: test_X, y_: test_y, keep_prob: 0.4}))

        saver.save(sess, 'model/my-model')


def restore_model():

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model/my-model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))


if __name__ == '__main__':
    # First run
    randomly_assign_train_test('images')
    run_tensorflow_model()

    # To restore model
    # restore_model()
