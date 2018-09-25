import os

import numpy as np

import util

CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

def setup_log_files(log_dir, blocks, options, mode='train'):
    """options should be a list of strings"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if 'gradient_norms' in options:
        filename = '{}_grad_norms.csv'.format(mode)
        with open(os.path.join(log_dir, filename), 'w') as csv:
            csv.write('pre_clip_grad_vals, post_clip_grad_vals,{}\n'.format(
                ','.join([
                    'pre_clip_wc_{}_grad_vals'.format(i)
                    for i in range(blocks)
                ])))
    if 'weak_classifier_covar' in options:
        filename = '{}_weak_learner_covariance.csv'.format(mode)
        with open(os.path.join(log_dir, filename),
                'w') as csv:
            csv.write('weak_classifier_covariance')
    dir = '{}_wc_confusion_matrices'.format(mode)
    if 'weak_classifier_conf_matrix' in options and not os.path.exists(
            os.path.join(log_dir, dir)):
        os.makedirs(os.path.join(log_dir, dir))

def process_metrics(log_dir, outs, options, epoch, mode='train'):
    """
    Note: this function does I/O with a number of files and stdout.
    """
    final_acc_val = outs['correct_final_prop']
    mean_final_acc_val = np.mean(final_acc_val)
    print("Final Accuracy: " + str(mean_final_acc_val))

    weak_acc_vals = outs['correct_weak_props']
    mean_weak_acc_vals = np.mean(weak_acc_vals, axis=0)
    print("Weak Classifier Accuracies: " + str(mean_weak_acc_vals))

    final_classification_loss_vals = outs['final_classification_loss']
    mean_final_classification_loss = np.mean(final_classification_loss_vals)
    print("Final classification loss: " + str(mean_final_classification_loss))

    if 'gradient_norms' in options:
        pre_clip_global_grad_vals = outs['pre_clip_global_grads']
        mean_pre_clip_global_grad_vals = np.mean(pre_clip_global_grad_vals)
        post_clip_global_grad_vals = outs['post_clip_global_grads']
        mean_post_clip_global_grad_vals = np.mean(
            post_clip_global_grad_vals)
        pre_clip_wc_grad_vals = outs['pre_clip_wc_grads']
        mean_pre_clip_wc_grad_vals = np.mean(pre_clip_wc_grad_vals, axis=0)
        filename = '{}_grad_norms.csv'.format(mode)
        with open(os.path.join(log_dir, filename),
                    'a') as csv:
            csv.write('{},{},{}\n'.format(
                mean_pre_clip_global_grad_vals,
                mean_post_clip_global_grad_vals, ','.join([
                    '{}'.format(val) for val in mean_pre_clip_wc_grad_vals
                ])))

    if 'weak_classifier_covar' in options:
        weak_classification_vals = np.reshape(
            np.swapaxes(outs['weak_classifiers'], 1, 2), (-1, 5, 10))
        mean_weak_classification_covar_vals = np.mean(
            np.array([
                np.cov(x, rowvar=False) for x in weak_classification_vals
            ]),
            axis=0)
        filename = '{}_weak_learner_covariance.csv'.format(mode)
        with open(
                os.path.join(log_dir, 'weak_learner_covariance.csv'),
                'a') as csv:
            csv.write('{}\n'.format(mean_weak_classification_covar_vals))

    # plots
    if 'weak_classifier_rate' in options:
        util.plot_parallel_histograms(
            values=np.swapaxes(weak_acc_vals, 0, 1),
            save_path=os.path.join(
                log_dir,
                '{}_weak_learner_classification_rate_epoch_{}'.format(mode, epoch)),
            hist_title='Weak Learner',
            xlabel='Correct Rate',
            ylabel='Minibatches')

    if 'weak_classifier_conf_matrix' in options:
        wc_conf_matrix_vals = np.sum(outs['wc_conf_matrices'], axis=0)
        [
            util.plot_confusion_matrix(
                cm,
                CIFAR10_CLASS_NAMES,
                title='{}_confusion_matrix_epoch_{}_wc_{}'.format(mode, epoch, i),
                log_dir=os.path.join(log_dir,
                                        '{}_wc_confusion_matrices'.format(mode)))
            for i, cm in enumerate(wc_conf_matrix_vals)
        ]