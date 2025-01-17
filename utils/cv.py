from sklearn.model_selection import BaseCrossValidator
import numpy as np

class GroupBasedCrossValidator(BaseCrossValidator):
    def __init__(self, n_splits=1, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        #TODO: add self.split_group list that tracks what the selected groups were for each split
        unique_groups = np.unique(groups)
        pos_groups = [grp for grp in unique_groups if grp.startswith('1_')]
        neg_groups = [grp for grp in unique_groups if grp.startswith('0_')]

        if not pos_groups or not neg_groups:
            raise ValueError("Both positive and negative groups should be present in the data")

        np.random.seed(self.random_state)
        for _ in range(self.n_splits):
            selected_pos_group = np.random.choice(pos_groups, 1)[0]
            selected_neg_group = np.random.choice(neg_groups, 1)[0]

            test_indices = np.where((groups == selected_pos_group) | (groups == selected_neg_group))[0]
            yield test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
