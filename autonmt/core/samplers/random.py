import torch
from torch.utils.data import Sampler


class RandomSampler(Sampler):
    """Random sampling with per-epoch reshuffle."""

    def __init__(self, data_source, seed=0):
        super().__init__()
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        yield from torch.randperm(len(self.data_source), generator=g).tolist()

    def __len__(self):
        return len(self.data_source)
