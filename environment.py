import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import loadData
from collections import defaultdict, deque

# from PIL import Image
# im = Image.fromarray(X_train[1])
# im.save('./data/0.png')

import matplotlib

# matplotlib.image.imsave('./data/1.png', X_train[0], cmap='gray')
# image_file = './data/1.png'
# image = plt.imread(image_file)
# plt.imshow(image)

# import scipy.misc
# scipy.misc.imsave('./data/0.png', X_train[0])

# print('')

def onehot(label):
    class_num = np.max(label).astype(int) + 1
    new_label = np.zeros((label.shape[0], class_num))
    new_label[np.arange(label.shape[0]), label.astype(int)] = 1
    return new_label

class DataEnv(object):

    X_train, Y_train, X_valid, Y_valid = loadData()

    # Y_train = onehot(Y_train)
    # Y_valid = onehot(Y_valid)

    action_dim = 2
    state_dim = 4
    label_set = np.unique(Y_train)
    num_classes = len(label_set)

    def __init__(self):
        self.X_train_features = self.get_features(self.X_train)
        self.X_valid_features = self.get_features(self.X_valid)

        # self.X_train = self.X_train
        # self.X_valid = self.X_valid

        self.seed_id = np.arange(10)
        self.unlabeled_id = np.arange(10, len(self.X_train))

        np.random.shuffle(self.unlabeled_id)

        self.seed_x = self.X_train[self.seed_id]
        self.seed_y = self.Y_train[self.seed_id]

        self.budgets = 10

    def compute_dist(self, sample_x_feature):
        nearest_dist = np.ones(self.num_classes) * 1e6
        all_dist = defaultdict(deque)
        for i,x in enumerate(self.labeled_x_features):
            dist = np.linalg.norm(x - sample_x_feature)
            c = int(self.labeled_y[i])
            all_dist[c].append(dist)
            if dist < nearest_dist[c]:
                nearest_dist[c] = dist
        return nearest_dist, all_dist

    def get_frame(self, classifier):
        sample_x = self.unlabeled_x[self.current_frame]
        sample_x_feature = classifier.getFeatures(sample_x)
        self.labeled_x_features = classifier.getFeatures(self.labeled_x)
        neighbor_dist, all_dist = self.compute_dist(sample_x_feature)
        predictions = classifier.getProb(sample_x)
        observation = np.hstack((neighbor_dist, predictions))
        return observation

    def get_labeled_data(self):
        if self.queried_set_x:
            self.labeled_x = np.concatenate((self.seed_x, np.vstack(self.queried_set_x)))
            self.labeled_y = np.concatenate((self.seed_y, np.vstack(self.queried_set_y)))
        else:
            self.labeled_x = self.seed_x
            self.labeled_y = self.seed_y

    def get_features(self, X_data):
        pass

    def step(self, action, classifier):
        is_terminal = False
        if action == 1:
            self.query()
            # if self.queried_time % 10 == 0:
            self.get_labeled_data()
            new_performance = classifier.get_performance(self.labeled_x, self.labeled_y, self.X_valid, self.Y_valid, new=True)
            reward = new_performance - self.performance
            if new_performance != self.performance:
                self.performance = new_performance
        else:
            reward = 0.
        if self.queried_time == self.budgets:
            is_terminal = True

        self.current_frame += 1
        next_observation = self.get_frame(classifier)

        return next_observation, reward, is_terminal


    def query(self):
        sample = self.unlabeled_x[self.current_frame]
        # simulate: obtain the labels
        label = self.unlabeled_y[self.current_frame]
        self.queried_time += 1
        # print "Select:", sentence, labels
        self.queried_set_x.append(sample)
        self.queried_set_y.append(label)

    def reset(self, classifier):
        np.random.shuffle(self.unlabeled_id)
        self.unlabeled_x = self.X_train[self.unlabeled_id]
        self.unlabeled_y = self.Y_train[self.unlabeled_id]

        self.queried_time = 0
        self.current_frame = 0
        self.performance = 0
        # select pool
        self.queried_set_x = deque()
        self.queried_set_y = deque()

        self.get_labeled_data()
        self.performance = classifier.get_performance(self.labeled_x, self.labeled_y, self.X_valid, self.Y_valid)
        sample_x = self.unlabeled_x[self.current_frame]
        sample_x_feature = classifier.getFeatures(sample_x)
        self.labeled_x_features = classifier.getFeatures(self.labeled_x)
        neighbor_dist, all_dist = self.compute_dist(sample_x_feature)
        predictions = classifier.getProb(sample_x)
        observation = np.hstack((neighbor_dist, predictions))
        return observation

    def render(self):
        pass

if __name__ == '__main__':
    env = DataEnv()

    print('')