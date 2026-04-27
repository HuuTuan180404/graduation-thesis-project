import logging
import numpy as np
from collections import Counter
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def logger(message=""):
    print(message)
    logging.info(message)


def __balance_val_split(dataset, val_split=0.0):
    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        targets = np.array(dataset.targets)

    train_indices, val_indices = train_test_split(
        np.arange(len(targets)), test_size=val_split, stratify=targets
    )

    if isinstance(dataset, Subset):
        train_dataset = Subset(
            dataset.dataset, np.array(dataset.indices)[train_indices]
        )
        val_dataset = Subset(dataset.dataset, np.array(dataset.indices)[val_indices])
    else:
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    

    return train_dataset, val_dataset


def __split_of_train_sequence(subset: Subset, train_split=1.0):
    if train_split == 1:
        return subset

    targets = np.array([subset.dataset.targets[i] for i in subset.indices])
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]), test_size=1 - train_split, stratify=targets
    )

    train_dataset = Subset(
        subset.dataset, indices=[subset.indices[i] for i in train_indices]
    )

    return train_dataset


def __log_class_statistics(subset: Subset):
    train_classes = [subset.dataset.targets[i] for i in subset.indices]
    print(dict(Counter(train_classes)))
