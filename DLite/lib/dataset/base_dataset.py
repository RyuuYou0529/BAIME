import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from ..core.register import DATASET_REGISTER

class BaseDataset(Dataset, ABC):
    def __init__(self):
        super(BaseDataset, self).__init__()
    
    @classmethod
    @abstractmethod
    def init_from_args(cls, args):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

@DATASET_REGISTER.register('ExampleRandomDataset')
class ExampleRandomDataset(BaseDataset):
    def __init__(self, data_path):
        super(ExampleRandomDataset, self).__init__()
        self.data_path = data_path
        self.dim = 1024
        self.size = 1000
        self.data = [torch.randn(self.dim) for _ in range(self.size)]
        self.labels = [torch.randn(32) for _ in range(self.size)]
    
    @classmethod
    def init_from_args(cls, args):
        instance = cls(args.train_data_path)
        return instance
    
    def __getitem__(self, index):
        item = self.data[index]
        label = self.labels[index]
        return item, label

    def __len__(self):
        return self.size