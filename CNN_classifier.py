import tensorflow as tf
import time
from collections import deque
from utils import loadData
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier(object):

    def __init__(self):
        self.sess = tf.Session()
        self.build_netword()
        self.sess.run(tf.global_variables_initializer())
        self.accHist = deque()
        self.lossHist = deque()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())
        self.accHist = deque()
        self.lossHist = deque()

    def build_netword(self):
        # build network
        epsilon = 1e-3

        self.x = tf.placeholder(tf.float32, [None, 32 * 32], name='input')
        self.y = tf.placeholder(tf.float32, [None, 1], name='label')
        input_reshape = tf.reshape(self.x, [-1, 32, 32, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_reshape,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=None,
            name='conv1')
        # batch_mean1, batch_var1 = tf.nn.moments(conv1, [0, 1, 2])
        # z1_hat = (conv1 - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        # scale1 = tf.Variable(tf.ones([32]))
        # beta1 = tf.Variable(tf.zeros([32]))
        # conv1 = scale1 * z1_hat + beta1
        conv1 = tf.nn.relu(conv1)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None,
            name='conv2')
        # batch_mean2, batch_var2 = tf.nn.moments(conv2, [0, 1, 2])
        # z2_hat = (conv2 - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
        # scale2 = tf.Variable(tf.ones([64]))
        # beta2 = tf.Variable(tf.zeros([64]))
        # conv2 = scale2 * z2_hat + beta2
        conv2 = tf.nn.relu(conv2)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64], name='flatten')

        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=None, name='dense1')
        # batch_mean3, batch_var3 = tf.nn.moments(dense1, [0])
        # z3_hat = (dense1 - batch_mean3) / tf.sqrt(batch_var3 + epsilon)
        # scale3 = tf.Variable(tf.ones([1024]))
        # beta3 = tf.Variable(tf.zeros([1024]))
        # dense1 = scale3 * z3_hat + beta3
        self.dense1 = tf.nn.relu(dense1)

        dropout1 = tf.layers.dropout(inputs=self.dense1, rate=0.5, name='dropout1')

        # dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=None, name='dense2')
        # batch_mean4, batch_var4 = tf.nn.moments(dense2,[0])
        # z4_hat = (dense2 - batch_mean4) / tf.sqrt(batch_var4 + epsilon)
        # scale4 = tf.Variable(tf.ones([512]))
        # beta4 = tf.Variable(tf.zeros([512]))
        # dense2 = scale4 * z4_hat + beta4
        # dense2 = tf.nn.relu(dense2)
        #
        # dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5)

        self.logits = tf.layers.dense(inputs=dropout1, units=1, name='logits')

        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=self.logits)
        self.prob = tf.nn.sigmoid(self.logits)
        self.predictions = tf.round(tf.nn.sigmoid(self.logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def getFeatures(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.sess.run(self.dense1, {self.x: x})

    def getProb(self, sample_x):
        sample_x = sample_x.reshape(1, -1)
        prob = np.zeros(2)
        cls, value = self.sess.run([self.predictions, tf.nn.sigmoid(self.logits)], {self.x: sample_x})
        prob[int(cls)] = value
        prob[1-int(cls)] = 1 - value
        return prob

    def getAllProb(self, X_data):
        return self.sess.run(self.prob, {self.x: X_data})

    def get_performance(self, X_train, Y_train, X_valid, Y_valid, new=False):

        accHist, lossHist = deque(), deque()
        start_time = time.time()
        t = 0
        for epoch in range(2):
            # training
            batchSize = 32
            for start, end in zip(range(0, len(X_train), batchSize), range(batchSize, len(X_train)+batchSize, batchSize)):
                if t % 10 == 0:
                    # testing
                    acc_valid, pred_valid, loss_valid = self.sess.run([self.accuracy, self.predictions, self.loss], {self.x: X_valid, self.y: Y_valid})
                    print("Validation: Step: %i" % t, "| Accurate: %.2f" % acc_valid, "| Loss: %.2f" % loss_valid, )
                    # print('')
                _, acc_, pred_, loss_ = self.sess.run([self.opt, self.accuracy, self.predictions, self.loss],
                                                 {self.x: X_train[start:end], self.y: Y_train[start:end]})
                # acc_valid, loss_valid = self.sess.run([self.accuracy, self.loss],
                #                                  {self.x: X_valid, self.y: Y_valid})
                accHist.append(acc_)
                lossHist.append(loss_)
                t += 1
        end_time = time.time()
        train_acc, train_loss = self.sess.run([self.accuracy, self.loss], {self.x: X_train, self.y: Y_train})
        final_acc, final_loss = self.sess.run([self.accuracy, self.loss], {self.x: X_valid, self.y: Y_valid})
        self.accHist.append(final_acc)
        self.lossHist.append(final_loss)
        # if new:
        # print('spent: %.4fs' % (end_time - start_time))
        print("Validation Accuracy: %.3f, Training Accuracy: %.3f" % (final_acc, train_acc))
        # print('--------------------------------------------------------------------------')

        # sns.lineplot(x=range(t), y=accHist, label='accuracy')
        # sns.lineplot(x=range(t), y=lossHist, label='loss')
        # plt.show()

        return final_acc


if __name__ == '__main__':
    X_train, Y_train, X_valid, Y_valid = loadData()
    clf = Classifier()
    # maxAcc = 0
    # minAcc = 1
    # for i in range(100):
    #     seed_id = np.random.choice(len(X_train), 100)
    #     X_train_new = X_train[seed_id]
    #     Y_train_new = Y_train[seed_id]
    #     clf.reset()
    #     acc = clf.get_performance(X_train_new, Y_train_new, X_valid, Y_valid)
    #     if acc > maxAcc:
    #         maxAcc = acc
    #     if acc < minAcc:
    #         minAcc = acc
    # print('maximal accuracy: ', maxAcc)
    # print('minimal accuracy: ', minAcc)

    acc = clf.get_performance(X_train, Y_train, X_valid, Y_valid)
