# coding:utf-8 
'''
created on 

@author:sunyihuan
'''

import tensorflow as tf
from practice_one.model.utils import *
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

BATCH_SIZE = 100


def preprocessing(trX, teX, trY, teY):
    res = RandomOverSampler(random_state=42)
    trY = np.argmax(trY, 1)
    teY = np.argmax(teY, 1)
    trX, trY = res.fit_sample(trX, trY)
    teX, teY = res.fit_sample(teX, teY)

    trY = np.eye(3)[trY]
    teY = np.eye(3)[teY]
    return trX, teX, trY, teY


def inference(input_tensor, avg_class, weights1, biases1):
    if avg_class == None:
        return tf.matmul(input_tensor, weights1) + biases1
    else:
        return tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)


def train(X_train, X_test, Y_train, Y_test, REGULARIZATION_RATE=0.0001, MOVING_AVERAGE_DECAY=0.99,
          LEARNING_RATE_BASE=0.8, LEARNING_RATE_DACAY=0.99, TRAINING_STEPS=2000):
    m, n_x = X_train.shape
    n_y = Y_train.shape[1]
    x = tf.placeholder(tf.float32, [None, n_x], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, n_y], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([n_x, n_y], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[n_y]))

    # weight2 = tf.Variable(tf.truncated_normal([LAYER1_NONE, n_y], stddev=0.1))
    # biases2 = tf.Variable(tf.constant(0.1, shape=[n_y]))

    y = inference(x, None, weights1, biases1)

    global_step = tf.Variable(0, trainable=False)

    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    average_y = inference(x, variables_averages, weights1, biases1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=1000, decay_rate=LEARNING_RATE_DACAY)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variables_averages_op)

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        validate_feed = {x: X_train, y_: Y_train}
        test_feed = {x: X_test, y_: Y_test}

        for i in range(TRAINING_STEPS):
            if i % 20 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),v acc is %g" % (i, validate_acc))
            sess.run(train_op, feed_dict=validate_feed)

        Y, test_acc = sess.run([y, accuracy], feed_dict=test_feed)
        print("After %d training step(s),test accuray is %g" % (TRAINING_STEPS, test_acc))
        Y = list(np.argmax(Y, 1))
        return Y


if __name__ == '__main__':
    file = "face_1_channel_sense"
    X_train, X_test, Y_train, Y_test = load_data(file)
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    LAYER1_NONE = 500
    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DACAY = 0.99
    REGULARIZATION_RATE = 0.001
    TRAINING_STEPS = 2000
    MOVING_AVERAGE_DECAY = 0.99
    Y = train(X_train, X_test, Y_train, Y_test, TRAINING_STEPS=TRAINING_STEPS)
    for i in range(3):
        print(str(i) + "比例", round(100 * Y.count(i) / len(Y), 2), "%")
    print(classification_report(y_true=np.argmax(Y_test, 1), y_pred=Y))