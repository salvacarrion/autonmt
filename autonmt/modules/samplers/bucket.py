import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np


class BucketIterator(Sampler):
    def __init__(self, data_source, batch_size, sort_key, shuffle=True, sort_within_batch=False):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort_key = sort_key
        self.sort_within_batch = sort_within_batch

        # Sort indices by the specified key (e.g., sequence length)
        self.sorted_indices = np.argsort([sort_key(x, y) for x, y in self.data_source])

        # Create buckets of indices
        self.buckets = [self.sorted_indices[i:i + batch_size] for i in range(0, len(self.sorted_indices), batch_size)]

    def __iter__(self):
        # Shuffle buckets if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())

            # Shuffle the list of buckets
            shuffled_indices = torch.randperm(len(self.buckets), generator=g).tolist()
            self.buckets = [self.buckets[i] for i in shuffled_indices]

        # Sort within each bucket if required
        if self.sort_within_batch:
            self.buckets = [sorted(bucket, key=lambda idx: self.sort_key(*self.data_source[idx]), reverse=True) for bucket in self.buckets]

        # Flatten the list of buckets into a list of indices
        indices = [idx for bucket in self.buckets for idx in bucket]
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
