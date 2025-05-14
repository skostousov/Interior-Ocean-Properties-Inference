from sklearn.model_selection import GroupShuffleSplit, GroupKFold

class NestedSplitter:
    def __init__(self, test_frac=0.2, outer_splits = 5, inner_splits = 5, seed=42):
        self.outer = GroupShuffleSplit(n_splits=outer_splits, test_size=test_frac, random_state=seed)
        self.inner = GroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    def __iter__(self):
        return self._generate_splits()
    def _generate_splits(self):
        for train_idx, test_idx in self.outer.split(self.indexes, groups=self.groups):
            for inner_tr, val in self.inner.split(train_idx):
                yield {
                    'train_idx': train_idx[inner_tr],
                    'val_idx' : train_idx[val],
                    'test_idx': test_idx
                }
    def split(self, dataset):
        self.dataset = dataset
        self.groups = self.dataset.index_region
        self.indexes = list(range(len(self.dataset)))
        return self