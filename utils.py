import os

import torch
import numpy as np
from scipy import sparse

from preprocess import load_sparse


def load_adj(path, device=torch.device('cuda')):
    filename = os.path.join(path, 'code_adj.npz')
    adj = torch.from_numpy(load_sparse(filename, return_sparse=False)).to(device=device, dtype=torch.float32)
    return adj


class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True,
                 device=torch.device('cuda'), cache_on_device=False):
        super().__init__()
        self.path = data_path
        self._load_data(label)
        
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.cache_on_device = cache_on_device
        if self.cache_on_device:
            self._cache_tensors_on_device()

    def _load_data(self, label):
        self.code_x = load_sparse(os.path.join(self.path, 'code_x.npz'), return_sparse=False)
        self._size = self.code_x.shape[0]
        self.visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        
        if label == 'm':
            self.y = load_sparse(os.path.join(self.path, 'code_y.npz'), return_sparse=False)
        elif label == 'h':
            self.y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        
        self.divided = load_sparse(os.path.join(self.path, 'divided.npz'), return_sparse=False)
        self.neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'), return_sparse=False)

    def _cache_tensors_on_device(self):
        print(f'Caching dataset on {self.device}: {self.path}')
        self.code_x = torch.as_tensor(self.code_x, device=self.device, dtype=torch.float32)
        self.visit_lens = torch.as_tensor(self.visit_lens, device=self.device, dtype=torch.long)
        self.y = torch.as_tensor(self.y, device=self.device, dtype=torch.float32)
        self.divided = torch.as_tensor(self.divided, device=self.device)
        self.neighbors = torch.as_tensor(self.neighbors, device=self.device)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        if torch.is_tensor(self.y):
            return self.y.detach().cpu().numpy()
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]

        if self.cache_on_device:
            if not torch.is_tensor(slices):
                slices = torch.as_tensor(slices, device=self.device, dtype=torch.long)
            return (
                self.code_x.index_select(0, slices),
                self.visit_lens.index_select(0, slices),
                self.divided.index_select(0, slices),
                self.y.index_select(0, slices),
                self.neighbors.index_select(0, slices),
            )
        
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        
        return code_x, visit_lens, divided, y, neighbors


class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i], )) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str
