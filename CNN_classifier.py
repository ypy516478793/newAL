import tensorflow as tf
import time
from collections import deque
from utils import loadData

class Classifier(object):

    def __init__(self):
        self.sess = tf.Session()
        self.build_netword()
        self.sess.run(tf.global_variables_initializer())


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
        batch_mean1, batch_var1 = tf.nn.moments(conv1, [0, 1, 2])
        z1_hat = (conv1 - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        scale1 = tf.Variable(tf.ones([32]))
        beta1 = tf.Variable(tf.zeros([32]))
        conv1 = scale1 * z1_hat + beta1
        conv1 = tf.nn.relu(conv1)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None,
            name='conv2')
        batch_mean2, batch_var2 = tf.nn.moments(conv2, [0, 1, 2])
        z2_hat = (conv2 - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
        scale2 = tf.Variable(tf.ones([64]))
        beta2 = tf.Variable(tf.zeros([64]))
        conv2 = scale2 * z2_hat + beta2
        conv2 = tf.nn.relu(conv2)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64], name='flatten')

        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=None, name='dense1')
        batch_mean3, batch_var3 = tf.nn.moments(dense1, [0])
        z3_hat = (dense1 - batch_mean3) / tf.sqrt(batch_var3 + epsilon)
        scale3 = tf.Variable(tf.ones([1024]))
        beta3 = tf.Variable(tf.zeros([1024]))
        dense1 = scale3 * z3_hat + beta3
        dense1 = tf.nn.relu(dense1)

        dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, name='dropout1')

        # dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=None, name='dense2')
        # batch_mean4, batch_var4 = tf.nn.moments(dense2,[0])
        # z4_hat = (dense2 - batch_mean4) / tf.sqrt(batch_var4 + epsilon)
        # scale4 = tf.Variable(tf.ones([512]))
        # beta4 = tf.Variable(tf.zeros([512]))
        # dense2 = scale4 * z4_hat + beta4
        # dense2 = tf.nn.relu(dense2)
        #
        # dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5)

        logits = tf.layers.dense(inputs=dropout1, units=1, name='logits')

        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=logits)

        self.predictions = tf.round(tf.nn.sigmoid(logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def get_performance(self, X_train, Y_train, X_valid, Y_valid):

        accHist, lossHist = [], []
        start_time = time.time()
        t = 0
        for epoch in range(1):
            # training
            batchSize = 32
            for start, end in zip(range(0, len(X_train), batchSize), range(batchSize, len(X_train)+batchSize, batchSize)):
                if t % 10 == 0:
                    # testing
                    acc_, pred_, loss_ = self.sess.run([self.accuracy, self.predictions, self.loss], {self.x: X_valid, self.y: Y_valid})
                    accuracies.append(acc_)
                    steps.append(t)
                    print("Validation: Step: %i" % t, "| Accurate: %.2f" % acc_, "| Loss: %.2f" % loss_, )
                    # print('')
                _, acc_, pred_, loss_ = self.sess.run([self.opt, self.accuracy, self.predictions, self.loss],
                                                 {self.x: X_train[start:end], self.y: Y_train[start:end]})
                t += 1
        end_time = time.time()
        print('spent: %.4fs' % (end_time - start_time))

        final_acc = self.sess.run(self.accuracy, {self.x: X_valid, self.y: Y_valid})
        print("Validation Accurate: %.2f" % final_acc)

        return final_acc


if __name__ == '__main__':
    X_train, Y_train, X_valid, Y_valid = loadData()
    clf = Classifier()
    clf.get_performance(X_train, Y_train, X_valid, Y_valid)


