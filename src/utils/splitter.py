from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
from torch.utils.data import Subset
import numpy as np
import torch
import os
import math

def simple_train_val_split(dataset, val_frac=0.2, seed=42):
     groups = dataset.indexed_region
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


def train_val_test_split_temp_strat(dataset, val_frac=0.1, test_frac=0.15, seed=42, test_indices_path = None):
    groups = dataset.groups
    month_of_year = [group % 12 for group in groups]

    n_splits_test = math.ceil(1/test_frac)
    n_splits_val = math.ceil(1/(val_frac/(1-test_frac)))

    all_indices = list(range(len(dataset)))

    if test_indices_path and os.path.exists(test_indices_path):
        test_idx = torch.load(test_indices_path, weights_only=False)
        print(f"Loaded existing test indices from {test_indices_path}, test size: {len(test_idx)}")
        
        train_val_mask = np.ones(len(dataset), dtype=bool)
        for idx in test_idx:
            train_val_mask[idx] = False

        remaining_indices = np.array(all_indices)[train_val_mask].tolist()
        remaining_groups = [groups[i] for i in remaining_indices]
        remaining_months = [month_of_year[i] for i in remaining_indices]

        gss_val = StratifiedGroupKFold(n_splits=n_splits_val, random_state=seed, shuffle=True)
        train_idx, val_idx = next(gss_val.split(remaining_indices, remaining_months, groups=remaining_groups))

        train_idx = [remaining_indices[i] for i in train_idx]
        val_idx = [remaining_indices[i] for i in val_idx]
    
    else:
        gss_test = StratifiedGroupKFold(n_splits=n_splits_test, random_state=seed, shuffle=True)
        train_val_idx, test_idx = next(gss_test.split(all_indices, month_of_year, groups=groups))

        gss_val = StratifiedGroupKFold(n_splits=n_splits_val, random_state=seed, shuffle=True)
        train_val_groups = [groups[i] for i in train_val_idx]
        train_val_months = [month_of_year[i] for i in train_val_idx]

        train_idx, val_idx = next(gss_val.split(train_val_idx, train_val_months, groups=train_val_groups))

        train_idx = [train_val_idx[i] for i in train_idx]
        val_idx = [train_val_idx[i] for i in val_idx]

        if test_indices_path:
            os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
            torch.save(test_idx, test_indices_path)
            print(f"Test indices saved to {test_indices_path}")
    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val overlap!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test overlap!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test overlap!"
    return train_idx, val_idx, test_idx

def train_val_test_split_temp(dataset, val_frac=0.1, test_frac=0.15, seed=42, test_indices_path = None, gen_new = False):
    print("entered splitter function")
    groups = dataset.groups

    all_indices = list(range(len(dataset)))

    if test_indices_path and os.path.exists(test_indices_path) and not gen_new:
        print(f"About to load existing test indices from {test_indices_path}")
        test_idx = torch.load(test_indices_path, weights_only=False)
        print(f"Loaded existing test indices from {test_indices_path}, test size: {len(test_idx)}")
        
        train_val_mask = np.ones(len(dataset), dtype=bool)
        for idx in test_idx:
            train_val_mask[idx] = False

        remaining_indices = np.array(all_indices)[train_val_mask].tolist()
        remaining_groups = [groups[i] for i in remaining_indices]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac/(1-test_frac), random_state=seed)
        train_idx, val_idx = next(gss_val.split(remaining_indices, groups=remaining_groups))

        train_idx = [remaining_indices[i] for i in train_idx]
        val_idx = [remaining_indices[i] for i in val_idx]
    
    else:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(all_indices, groups=groups))

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1-test_frac), random_state=seed)
        train_val_groups = [groups[i] for i in train_val_idx]

        train_idx, val_idx = next(gss_val.split(train_val_idx, groups=train_val_groups))

        train_idx = [train_val_idx[i] for i in train_idx]
        val_idx = [train_val_idx[i] for i in val_idx]

        if test_indices_path:
            os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
            torch.save(test_idx, test_indices_path)
            print(f"Test indices saved to {test_indices_path}")
    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val overlap!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test overlap!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test overlap!"
    return train_idx, val_idx, test_idx

def train_val_test_split(dataset, val_frac=0.15, test_frac=0.15, seed=42, test_indices_path = None, groups=None,):
    groups = dataset.indexed_region
    all_indices = list(range(len(dataset)))
    

    if test_indices_path and os.path.exists(test_indices_path):
        test_idx = torch.load(test_indices_path, weights_only=False)
        print(f"Loaded existing test indices from {test_indices_path}, test size: {len(test_idx)}")
        
        train_val_mask = np.ones(len(dataset), dtype=bool)
        for idx in test_idx:
            train_val_mask[idx] = False

        remaining_indices = np.array(all_indices)[train_val_mask].tolist()
        remaining_groups = [groups[i] for i in remaining_indices]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac/(1-test_frac), random_state=seed)
        train_idx, val_idx = next(gss_val.split(remaining_indices, groups=remaining_groups))

        train_idx = [remaining_indices[i] for i in train_idx]
        val_idx = [remaining_indices[i] for i in val_idx]
    else:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(all_indices, groups=groups))

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1-test_frac), random_state=seed)
        train_val_groups = [groups[i] for i in train_val_idx]

        train_idx, val_idx = next(gss_val.split(train_val_idx, groups=train_val_groups))

        train_idx = [train_val_idx[i] for i in train_idx]
        val_idx = [train_val_idx[i] for i in val_idx]

        if test_indices_path:
            os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
            torch.save(test_idx, test_indices_path)
            print(f"Test indices saved to {test_indices_path}")

    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val overlap!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test overlap!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test overlap!"

    return train_idx, val_idx, test_idx




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
    
def test_indices(test_indices_path):
    test_idx = torch.load(test_indices_path, weights_only=False)
    print(f"Loaded existing test indices from {test_indices_path}, test size: {len(test_idx)}")
    return test_idx

if __name__ == "__main__":
    from src.utils.alternate_dataset import myDataset
    dataset = myDataset(season="Summer")
    train_idx, val_idx, test_idx = train_val_test_split_temp(dataset, gen_new=True)
    print(train_idx)