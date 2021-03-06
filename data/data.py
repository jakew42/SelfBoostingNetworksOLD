import os

import numpy as np

import data.mnist as mnist
import data.cifar10 as cifar10
import data.util as util


def data_generator(array, batch_size):
    def inf_train_gen():
        while True:
            np.random.shuffle(array)
            for i in range(0, len(array) - batch_size + 1, batch_size):
                yield np.array(array[i:i + batch_size])

    return inf_train_gen()


def parallel_data_generator(arrays, batch_size):
    if not hasattr(arrays, '__iter__'): arrays = [arrays]

    def unison_shuffled_copies(arrays):
        assert all([len(a) == len(arrays[0]) for a in arrays])
        p = np.random.permutation(len(arrays[0]))
        return [a[p] for a in arrays]

    def inf_train_gen(arrays):
        while True:
            arrays = unison_shuffled_copies(arrays)
            for i in range(0, len(arrays[0]) - batch_size + 1, batch_size):
                yield [np.array(a[i:i + batch_size]) for a in arrays]

    return inf_train_gen(arrays)


def load_cifar10(data_dir):
    """Returns CIFAR10 as (train_data, train_labels, test_data, test_labels)
    
    Shapes are (50000, 32, 32, 3), (50000, 10), (10000, 32, 32, 3), (10000, 10)
    Data is in [0,1] and labels are one-hot
    """
    if not os.path.exists(os.path.join(data_dir, 'cifar10_train_data.npy')):
        cifar10.download_and_extract_npy(data_dir)

    train_data = np.load(os.path.join(
        data_dir, 'cifar10_train_data.npy')).astype('float32') / 255.0
    train_labels = np.load(os.path.join(data_dir, 'cifar10_train_labels.npy'))
    train_labels = util.one_hot(train_labels, 10)
    test_data = np.load(os.path.join(
        data_dir, 'cifar10_test_data.npy')).astype('float32') / 255.0
    test_labels = np.load(os.path.join(data_dir, 'cifar10_test_labels.npy'))
    test_labels = util.one_hot(test_labels, 10)
    return train_data, train_labels, test_data, test_labels


def load_mnist(data_dir):
    train_labels = util.one_hot(mnist.train_labels(), 10)
    test_labels = util.one_hot(mnist.test_labels(), 10)
    return mnist.train_images(), train_labels, mnist.test_images(), test_labels


def filter_classes(data, labels, classes):
    class_num = len(classes)
    remapping = dict(zip(classes, range(len(classes))))
    remapping_fn = lambda x: remapping[x]

    dense_labels = np.argmax(labels, axis=1)
    idx_list = np.where(np.isin(dense_labels, classes))
    data = data[idx_list]
    labels = labels[idx_list]
    filtered_dense_labels = np.argmax(labels, axis=1)
    remapped_labels = list(map(remapping_fn, filtered_dense_labels))
    labels = np.eye(class_num)[remapped_labels]

    return data, labels

def load_data(dataset_name, batch_size, classes=None):
    # load the data
    if dataset_name == 'cifar10':
        train_data, train_labels, test_data, test_labels = load_cifar10(
            './data/')
    elif dataset_name == 'mnist':
        train_data, train_labels, test_data, test_labels = load_mnist(
            './data/')

    # Use all classes if the default value of [] is present
    if classes:
        train_data, train_labels = filter_classes(train_data, train_labels, classes)
        test_data, test_labels = filter_classes(test_data, test_labels, classes)

    train_data_shape = train_data.shape
    test_data_shape = test_data.shape
    label_shape = train_labels.shape
    class_num = train_labels.shape[1]
    train_gen = parallel_data_generator([train_data, train_labels], batch_size)
    test_gen = parallel_data_generator([test_data, test_labels], batch_size)
    return train_gen, test_gen, train_data_shape, test_data_shape, label_shape, class_num
