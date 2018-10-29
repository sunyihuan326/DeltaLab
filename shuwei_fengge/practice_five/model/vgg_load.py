# coding:utf-8
'''
Created on 2018/1/15.

@author: chk01
'''
from practice_five.model.nst_utils import *
from tensorflow.python.framework import ops


def preprocessing(trX, teX, trY, teY):
    # m, _ = trX.shape
    # res = SMOTE(ratio={0: int(1.8 * 0.25 * m), 1: int(1.8 * .60 * m), 2: int(1.8 * .15 * m)})
    # trX, trY = res.fit_sample(trX, np.argmax(trY, 1))
    # trY = np.eye(3)[trY]

    trX = trX / 255.
    teX = teX / 255.

    return trX, teX, trY, teY


def model(X_train, X_test, Y_train, Y_test, epochs=300, minibatch_size=64,
          initial_learning_rate=1.0, minest_learning_rate=0.001):
    ops.reset_default_graph()
    m, n_x = X_train.shape
    n_y = Y_train.shape[1]
    global_step = tf.Variable(0, trainable=False)

    model = load_vgg_model("F:/AImodel/imagenet-vgg-verydeep-19.mat")

    # X = tf.placeholder(dtype=tf.float32, shape=(None, 64 * 64 * 3))
    X = model['input']
    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y))

    vgg_layer = model['conv5_4']
    convZ = tf.contrib.layers.flatten(vgg_layer)
    # Z1 = tf.layers.dense(convZ, 4 * n_y, activation=tf.nn.relu)
    # Z2 = tf.layers.dense(Z1, 2 * n_y, activation=tf.nn.relu)
    ZL = tf.layers.dense(convZ, n_y, activation=None, name='output')

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=100, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, minest_learning_rate)
    loss = tf.reduce_mean(tf.square((Y - ZL)))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            for minibatch_X, minibatch_Y in minibatches(X_train, Y_train, minibatch_size, shuffle=True):
                __, _loss, _, res, llr = sess.run([add_global, loss, train_op, ZL, learning_rate],
                                                  feed_dict={X: minibatch_X.reshape(-1, 64, 64, 3), Y: minibatch_Y})
                minibatch_cost += _loss / num_minibatches

            if epoch % 10 == 0:
                test_loss = loss.eval(feed_dict={X: X_test.reshape(-1, 64, 64, 3), Y: Y_test})
                print('epoch', epoch, 'loss', minibatch_cost)
                print(llr)
                print(test_loss)

        saver.save(sess, "save/model-fc3-{}-{}.ckpt".format(epochs, int(test_loss)))
    return True


if __name__ == '__main__':
    file = '../data/face_top_9.mat'
    data = scio.loadmat(file)

    # load data
    X_train_org, X_test_org, Y_train_org, Y_test_org = load_data(file, test_size=0.2)
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train_org, X_test_org, Y_train_org, Y_test_org)
    model(X_train, X_test, Y_train[:, 6:12], Y_test[:, 6:12], epochs=500)
