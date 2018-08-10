import os

import sacred

import voting_strategy
import boosting_strategy
import networks.classifiers as classifiers
import networks.stems as stems
import networks.blocks as blocks
import train_boosted

ex = sacred.Experiment('boosted_experiment')


@ex.config
def cfg():
    stem_name = 'BigConvStem'
    block_name = 'IdentityBlock'
    classifier_name = 'ReduceFlattenClassifier'
    voting_strategy_name = 'samme.r'
    boosting_strategy_name = 'samme.r'
    training_style = ''
    epochs = 100
    batch_size = 32
    block_num = 10
    dataset_name = 'cifar10'
    # all if empty, otherwise filter to specified class #'s
    classes = []
    # 'gradient_norms', 'weak_classifier_covar', 'weak_classifier_rate', 'weak_classifier_conf_matrix'
    metrics_options = []
    log_dir = './logs/'
    load_stem = False
    class_num = 10

if 'SLURM_JOB_ID' in os.environ.keys():
    sacred.SETTINGS['CAPTURE_MODE'] = 'no'
    job_id = os.environ['SLURM_JOB_ID']
    obs_name = 'boosted_results_{}'.format(job_id)
else:
    obs_name = 'boosted_results'
ex.observers.append(sacred.observers.TinyDbObserver.create(obs_name))


@ex.automain
def run(stem_name, block_name, classifier_name, voting_strategy_name,
        boosting_strategy_name, training_style, epochs, batch_size, block_num,
        dataset_name, classes, metrics_options, log_dir, load_stem, class_num):
    # command line option dictionaries
    block_dict = {
        'IdentityBlock': blocks.IdentityBlock,
        'ResidualConvBlock': blocks.ResidualConvBlock
    }

    classifier_dict = {
        'ReduceFlattenClassifier': classifiers.ReduceFlattenClassifier
    }
    stem_dict = {'BigConvStem': stems.BigConvStem}

    voting_strategy_dict = {
        'naive':
        voting_strategy.naive_voting_strategy,
        'linear_combo':
        voting_strategy.LinearComboStrategy(block_num),
        'mlp_combo':
        voting_strategy.MLPComboStrategy([50, 20, class_num]),
        'samme.r':
        voting_strategy.SAMME_R_voting_strategy,
        'jake_experimental':
        voting_strategy.JakeExperimentalVotingStrategy(class_num, block_num)
    }

    boosting_strategy_dict = {
        'naive': boosting_strategy.non_boosting_strategy,
        'samme.r': boosting_strategy.SAMME_R_boosting_strategy
    }

    train_boosted.run(
        stem_fn=stem_dict[stem_name],
        block_fn=block_dict[block_name],
        classifier_fn=classifier_dict[classifier_name],
        voting_strategy_fn=voting_strategy_dict[voting_strategy_name],
        boosting_strategy_fn=boosting_strategy_dict[boosting_strategy_name],
        training_style=training_style,
        epochs=epochs,
        batch_size=batch_size,
        block_num=block_num,
        dataset_name=dataset_name,
        classes=classes,
        metrics_options=metrics_options,
        log_dir=log_dir,
        load_stem=load_stem)
