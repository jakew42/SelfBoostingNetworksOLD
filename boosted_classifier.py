import tensorflow as tf
import sonnet as snt

import networks


class BoostedClassifier(snt.AbstractModule):
    """
    Classifier module which performs self-boosting using a provided network,
    voting strategy, and boosting strategy. (TODO: the latter two)
    """

    def __init__(self,
                 voting_strategy,
                 blocks,
                 classifiers,
                 class_num,
                 name='boosted_classifier'):
        """
        Args:
          voting_strategy: A callable which takes a list of logits and returns
                           the final, boosted classification
          blocks: A list of modules, applied in succession after stem
          classifiers: A list parallel to blocks, to be weak learners
        """
        super(BoostedClassifier, self).__init__(name=name)
        self.voting_strategy = voting_strategy
        assert len(blocks) == len(
            classifiers), 'Must have equal number of blocks and classifiers'
        self._blocks = blocks
        self._classifiers = classifiers
        self._class_num = class_num

    def _build(self, inputs):
        x = inputs
        logits = []
        for i, _ in enumerate(self._blocks):
            x = self._blocks[i](x)
            c = self._classifiers[i](x)
            logits.append(c)

        final_classification = self.voting_strategy(logits)
        return final_classification, logits
