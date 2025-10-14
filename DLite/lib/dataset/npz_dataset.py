import numpy as np
import torch

from .base_dataset import BaseDataset
from ..core.register import DATASET_REGISTER

@DATASET_REGISTER.register('NPZDataset')
class NPZDataset(BaseDataset):
    def __init__(self, data_path):
        super(NPZDataset, self).__init__()
        self.data_path = data_path
        self.npz_file = np.load(self.data_path)
        self.data = self.npz_file['X']
        self.label = self.npz_file['Y']

    @classmethod
    def init_from_args(cls, args):
        instance = cls(args.train_data_path)
        return instance
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_np = self.data[index]
        data_tensor = torch.from_numpy(data_np).to(torch.float32)
        label_np = self.label[index]
        label_tensor = torch.from_numpy(label_np).to(torch.float32)
        return data_tensor, label_tensor