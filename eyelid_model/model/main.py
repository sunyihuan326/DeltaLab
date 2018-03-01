# coding:utf-8
'''
Created on 2018/2/10.

@author: Dxq
'''
import os
import tensorflow as tf

import numpy as np
from tqdm import tqdm

from eyelid_model.model.config import cfg
from eyelid_model.model.utils import load_data, get_batch_data
from eyelid_model.model.Net import DxqNet


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return (fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return (fd_test_acc)


def train(model, supervisor):
    fd_train_acc, fd_loss, fd_val_acc = save_to()
    num_tr_batch = 2 * 2850 // cfg.batch_size
    num_val_batch = 2 * 150 // cfg.batch_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch + 1, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for __ in tqdm(range(num_tr_batch), ncols=70, leave=False, unit='b'):
                sess.run(model.train_op)
                global_step = sess.run(model.global_step)
                if global_step % cfg.train_sum_freq == 0:
                    loss, train_acc, summary_str = sess.run([model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc) + "\n")
                    fd_train_acc.flush()

                if cfg.val_sum_freq != 0 and (global_step % cfg.val_sum_freq == 0):
                    val_acc = 0
                    valX, vallabels = sess.run([model.valX, model.vallabels])
                    for i in range(num_val_batch):
                        acc = sess.run(model.accuracy, {model.X: valX, model.labels: vallabels})
                        val_acc += acc
                    val_acc = val_acc / num_val_batch
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            coord.request_stop()
            coord.join(threads)
            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    fd_val_acc.close()
    fd_train_acc.close()
    fd_loss.close()


def evaluation():
    num_te_batch = 150 // cfg.batch_size
    fd_test_acc = save_to()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(cfg.logdir + '/model_epoch_0009_step_890.meta')
        saver.restore(sess, cfg.logdir + '/model_epoch_0009_step_890')
        graph = tf.get_default_graph()

        teX, teY = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads, train_mode='test', graph=sess.graph)

        tf.logging.info('Model restored!')
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        test_acc = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess, start=True)

        for i in tqdm(range(num_te_batch), ncols=70, leave=False, unit='b'):
            # print(tf.train.latest_checkpoint('./'))
            X = graph.get_tensor_by_name('shuffle_batch:0')
            labels = graph.get_tensor_by_name('shuffle_batch:1')
            accuracy = graph.get_tensor_by_name('acc:0')
            tteX, tteY = sess.run([teX, teY])
            acc = sess.run(accuracy, {X: tteX, labels: tteY})
            test_acc += acc

        test_acc = test_acc / num_te_batch
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

        coord.request_stop()
        coord.join(threads)


def main(_):
    tf.logging.info('Loading Graph...')
    model = DxqNet()
    tf.logging.info('Graph loaded ' + str(model.graph))
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, sv)
        tf.logging.info('Training done')
    else:
        evaluation()


if __name__ == "__main__":
    tf.app.run()
