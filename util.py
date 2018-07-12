import os

import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def run_epoch_ops(session,
                  steps_per_epoch,
                  verbose_ops=[],
                  silent_ops=[],
                  feed_dict_fn=lambda: None,
                  verbose=False):
    """
    Args:
        - session: tf.Session
        - steps_per_epoch: (int)
        - verbose_ops: ({str: tf.Tensor})
        - feed_dict_fn (callable): called to retrieve the feed_dict
                                   (dict of placeholders to np arrays)
        - verbose (bool): whether to use tqdm progressbar on stdout

    Return:
        Dict of str to numpy arrays or floats
    """
    verbose_vals = {k: [] for k, v in verbose_ops.items()}
    iterable = range(steps_per_epoch)
    if verbose: iterable = tqdm(iterable)

    for _ in iterable:
        out = session.run(
           [silent_ops, verbose_ops], feed_dict=feed_dict_fn())[1]
        verbose_vals = {k: v + [out[k]] for k, v in verbose_vals.items()}

    return {k: np.stack(v) for k, v in verbose_vals.items()}


def plot_parallel_histograms(values, save_path, hist_title, xlabel, ylabel):
    fig, axs = plt.subplots(
        1,
        len(values),
        sharey=True,
        tight_layout=True,
        figsize=(4.8 * len(values), 6.4))
    [
        histogram(
            values[i],
            axs[i],
            num_bins=100,
            title=hist_title + ' {}'.format(i),
            xlabel=xlabel,
            ylabel=ylabel) for i in range(len(values))
    ]
    fig.savefig(save_path)


def histogram(x, ax, num_bins, title, xlabel, ylabel):
    """
    Args:
        - x: numpy array of data to plot
        - ax: matplotlib axis to plot histogram on
        - num_bins: pass through to matplotlib ax.hist
        - title: str to put above histogram
        - xlabel: str to put on x-axis
        - ylabel: str to put on y-axis
    """
    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, density=1, range=[0, 1])

    mu = np.mean(x)
    sigma = np.std(x)
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma *
                                                             (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('{}: $\mu={:.3f}$, $\sigma={:.3f}$'.format(title, mu, sigma))

def plot_confusion_matrix(cm, classes,
                          title,
                          log_dir,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(log_dir, title + '.png'))
    plt.clf()
