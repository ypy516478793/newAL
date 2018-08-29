import numpy as np
import h5py
import random
import scipy.ndimage
import matplotlib.pyplot as plt

def loadData():

    data = h5py.File('./data/Lung_Nodule_2d.h5', 'r')
    X_train = data['X_train'][:]
    Y_train = data['Y_train'][:]
    X_valid = data['X_valid'][:]
    Y_valid = data['Y_valid'][:]
    data.close()

    # X_train_ratate = random_rotation_2d(X_train, 180)
    # X_train_noise = add_noise(X_train)
    # X_train = np.concatenate([X_train, X_train_ratate, X_train_noise], axis=0)
    # Y_train = np.tile(Y_train, 3)

    X_train = np.reshape(X_train, (-1, X_train.shape[1] * X_train.shape[2]))
    X_valid = np.reshape(X_valid, (-1, X_valid.shape[1] * X_valid.shape[2]))

    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    X_train = (X_train - mean_train) / std_train

    mean_valid = np.mean(X_valid, axis=0)
    std_valid = np.std(X_valid, axis=0)
    X_valid = (X_valid - mean_valid) / std_valid

    Y_train = Y_train[:, np.newaxis]
    Y_valid = Y_valid[:, np.newaxis]

    return X_train, Y_train, X_valid, Y_valid

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
    X_train, Y_train, X_valid, Y_valid = loadData()
    X_train = X_train.reshape(-1, 32 ,32)
    X_new = add_noise(X_train[:5], 180)
    print('rotated image: \n')
    plt.figure()
    plt.imshow(X_new[0])
    plt.show()
    print("")