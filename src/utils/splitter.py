from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from torch.utils.data import Subset

def simple_train_val_split(dataset, val_frac=0.2, seed=42):
     groups = dataset.index_region
     print(len(groups))
     print(len(dataset))
     indexes = list(range(len(dataset)))
     gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
     train_idx, val_idx = next(gss.split(indexes, groups=groups))
     print(f"Max train_idx: {max(train_idx)}, Max val_idx: {max(val_idx)}, Dataset size: {len(dataset)}")
     assert max(train_idx) < len(dataset), f"Max train_idx ({max(train_idx)}) >= dataset size ({len(dataset)})"
     assert max(val_idx) < len(dataset), f"Max val_idx ({max(val_idx)}) >= dataset size ({len(dataset)})"
     return (
         Subset(dataset, train_idx),
         Subset(dataset, val_idx)
     )

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