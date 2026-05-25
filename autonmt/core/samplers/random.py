import torch
from torch.utils.data import Sampler


class RandomIterator(Sampler):
    def __init__(self, data_source):
        super().__init__()
        self.data_source = data_source

    def __iter__(self):
        # Get the current random seed
        g = torch.Generator()
        g.manual_seed(torch.initial_seed())

        # Shuffle indices based on the current PyTorch seed
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)