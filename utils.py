import numpy as np
import h5py
import random
import scipy.ndimage
import matplotlib.pyplot as plt

def normal(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def reformat(x, y, img_size, num_ch, num_class):
    """ Reformats the data to the format acceptable for the conv layers"""
    dataset = x.reshape(
        (-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels

def loadData(dim=2, normalize='standard', one_hot=False):
    if dim == 2:
        h5f = h5py.File('./data/Lung_Nodule_2d.h5', 'r')
        X_train = h5f['X_train'][:]
        Y_train = h5f['Y_train'][:]
        X_valid = h5f['X_valid'][:]
        Y_valid = h5f['Y_valid'][:]
        h5f.close()

    image_size, num_classes, num_channels = X_train.shape[1], len(np.unique(Y_train)), 1
    X_train = np.maximum(np.minimum(X_train, 4096.), 0.)
    X_valid = np.maximum(np.minimum(X_valid, 4096.), 0.)

    if normalize == 'standard':
        mt = np.mean(X_train, axis=0)
        st = np.std(X_train, axis=0)
        X_train = (X_train - mt) / st
        mv= np.mean(X_valid, axis=0)
        sv = np.std(X_valid, axis=0)
        X_valid = (X_valid - mv) / sv
    elif normalize == 'unity_based':
        X_train = np.asanyarray([normal(X_train[i]) for i in range(len(X_train))])
        X_valid = np.asanyarray([normal(X_valid[i]) for i in range(len(X_valid))])

    if one_hot:
        X_train, Y_train = reformat(X_train, Y_train, image_size, num_channels, num_classes)
        X_valid, Y_valid = reformat(X_valid, Y_valid, image_size, num_channels, num_classes)
    elif not one_hot:
        X_train, _ = reformat(X_train, Y_train, image_size, num_channels, num_classes)
        X_valid, _ = reformat(X_valid, Y_valid, image_size, num_channels, num_classes)

    X_train = X_train.reshape(-1, 32*32)
    X_valid = X_valid.reshape(-1, 32*32)
    Y_train = Y_train.reshape(-1, 1)
    Y_valid = Y_valid.reshape(-1, 1)

    return X_train, Y_train, X_valid, Y_valid

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def precision_recall(y_true, y_pred):
    """
    Computes the precision and recall values for the positive class
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    TP = FP = FN = TN = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
    precision = (TP * 100.0) / (TP + FP)
    recall = (TP * 100.0) / (TP + FN)
    print('Precision: {0:.2f}'.format(precision))
    print('Recall: {0:.2f}'.format(recall))

def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        image = np.squeeze(batch[i])
        angle = random.uniform(-max_angle, max_angle)
        batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
    return batch_rot.reshape(size)

def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)

if __name__ == '__main__':
    X_train, Y_train, X_valid, Y_valid = loadData(one_hot=False)
    X_new = add_noise(X_train[:5], 180)
    print('noisy image: \n')
    plt.figure()
    plt.imshow(X_new[0])
    plt.show()
    print("")