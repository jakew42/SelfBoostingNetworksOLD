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
        - ops: ({str: tf.Tensor})
        - steps_per_epoch: (int)

    Return:
        Dict of str to numpy arrays or floats
    """
    epoch_vals = [0] * len(verbose_ops)
    if verbose:
        iterable = tqdm(list(range(steps_per_epoch)))
    else:
        iterable = list(range(steps_per_epoch))
    for i in iterable:
        step_vals = session.run([silent_ops, verbose_ops], feed_dict=feed_dict_fn())[1]
        epoch_vals = [
            i_x[1] + step_vals[i_x[0]] for i_x in enumerate(epoch_vals)
        ]
    return [x / float(steps_per_epoch) for x in epoch_vals]