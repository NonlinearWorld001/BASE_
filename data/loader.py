import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import torch.distributed as dist

class DynamicBatchSampler(Sampler):
  def __init__(
      self, 
      dataset: Dataset,
      batch_size: int,
      batch_size_factor: int=1,
      shuffle: bool=True,
      drop_last: bool=False,
      seed: int=42,
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.batch_size_factor = batch_size_factor
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.seed = seed
    self.epoch = 0

    self.effective_batch_size = batch_size * batch_size_factor # real batch size
    self.indices = list(range(len(dataset))) # initialize indices
  
  def set_epoch(
      self,
      epoch: int,
  ):
    '''
    set epoch for distrivbuted training
    '''
    self.epoch = epoch

  def __iter__(self) -> Iterator[List[int]]:
    '''
    generate batch indices
    '''
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    if self.shuffle:
      indices = torch.randperm(len(self.dataset), generator=g).tolist()
    else:
      indices = list(range(len(self.dataset)))

    # generate batch
    batch = []
    for idx in indices:
      batch.append(idx)
      if len(batch) == self.effective_batch_size:
        yield batch
        batch = []
    
    if len(batch) > 0 and not self.drop_last:
      yield batch

  def __len__(self) -> int:
    '''
    return batch number
    '''
    if self.drop_last:
      return len(self.dataset) // self.effective_batch_size
    else:
      return (len(self.dataset) + self.effective_batch_size - 1) // self.effective_batch_size


class DataLoader:
  def __init__(
      self,
      dataset: Dataset,
      batch_size: int=64,
      batch_size_factor: int=1,
      shuffle: bool=True, 
      num_workers: int=4,
      pin_memory: bool=True,
      drop_last: bool=False,
      collate_fn: Optional[Callable]=None,
      distributed: bool=False,
      seed: int=42,
      prefetch_factor: int=2,
      persistent_workers: bool=True,
  ):
    '''
    Initialize DataLoader
        
        Args:
            dataset: Target dataset for loading samples
            batch_size: Basic batch size for training
            batch_size_factor: Scaling factor of batch size, actual batch size = batch_size * batch_size_factor
            shuffle: Whether to shuffle the dataset indices
            num_workers: Number of subprocesses for parallel data loading
            pin_memory: Whether to pin tensor in CUDA pinned memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Custom function to merge a list of samples into a batch tensor
            distributed: Whether to enable distributed training mode; must set False while use pytorch_lightning
            seed: Random seed for reproducible data sampling
            prefetch_factor: Number of batches to prefetch for each worker process
            persistent_workers: Whether to keep worker subprocesses alive across epochs
    '''
    self.dataset = dataset
    self.batch_size = batch_size
    self.batch_size_factor = batch_size_factor
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.drop_last = drop_last
    self.collate_fn = collate_fn or partial(self.default_collate_fn)
    self.distributed = distributed
    self.seed = seed
    self.prefetch_factor = prefetch_factor
    self.persistent_workers = persistent_workers

    # calculate real batch size
    self.effective_batch_size = batch_size * batch_size_factor

    # initialize sampler
    self._init_sampler()

    # initialize pytorch dataloader
    self._init_torch_dataloader()

    # iterator status
    self.epoch = 0
    self.iter = 0

  def _init_sampler(self):
    '''
    initialize sampler
    '''
    if self.distributed and dist.is_available() and dist.is_initialized():
      self.sampler = DistributedSampler(
        self.dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=self.shuffle,
        seed=self.seed,
        drop_last=self.drop_last,
      )
    # dist.get_world_size() 返回当前可用的训练进程总数（即GPU数量）
    # dist.get_rank() 返回GUP的编号
    else:
      self.sampler = DynamicBatchSampler(
                dataset=self.dataset,
                batch_size=self.batch_size,
                batch_size_factor=self.batch_size_factor,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                seed=self.seed,
            )
      
  def _init_torch_dataloader(self):
    '''
    initial pytorch dataloader
    '''
    if self.distributed and dist.is_available() and dist.is_initialized():
      self.torch_dataloader = TorchDataLoader(
        dataset=self.dataset,
        batch_size=self.effective_batch_size,
        sampler=self.sampler,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        collate_fn=self.collate_fn,
        prefetch_factor=self.prefetch_factor,
        persistent_workers=self.persistent_workers and self.num_workers > 0,
      )
    else:
      self.torch_dataloader = TorchDataLoader(
        dataset=self.dataset,
        batch_sampler=self.sampler,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        collate_fn=self.collate_fn,
        prefetch_factor=self.prefetch_factor,
        persistent_workers= self.persistent_workers and self.num_workers > 0,
      )
  
  @staticmethod
  def default_collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
      batch: batch data list, every single element: (data, target/label)
    '''
    data, targets = zip(*batch)

    if isinstance(data[0], torch.Tensor):
      data = torch.stack(data, dim=0)
    elif isinstance(data[0], np.ndarray):
      data = torch.stack([torch.from_numpy(d) for d in data])
    else:
      data = torch.tensor(data)
    
    if isinstance(targets[0], torch.Tensor):
      targets = torch.stack(targets, dim=0)
    elif isinstance(targets[0], np.ndarray):
      targets = torch.stack([torch.from_numpy(t) for t in targets], dim=0)
    else:
      targets = torch.tensor(targets)
    
    return data, targets
  
  def set_epoch(
      self,
      epoch: int,
  ):
    '''
    set current epoch
    Args:
      epoch: current epoch
    '''
    self.epoch = epoch
    if hasattr(self.sampler, 'set_epoch'):
      self.sampler.set_epoch(epoch=epoch)

  def __iter__(self) -> Iterator:
    self.iter = 0
    return iter(self.torch_dataloader)
  
  def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    get next batch
    '''
    if not hasattr(self, '_iterator'):
        self._iterator = iter(self)
    try: 
      batch = next(self._iterator)
      self.iter += 1
      return batch
    except StopIteration:
      self._iterator = iter(self)
      raise StopIteration
    
  def __len__(self) -> int:
    return len(self.torch_dataloader)
  
  def get_batch_info(self)-> dict[str, Any]:
    return {
      'effective_batch_size': self.effective_batch_size,
      'num_batches': len(self),
      'dataset_size': len(self.dataset),
      'epoch': self.epoch,
      'iter_count': self.iter,
      'shuffle': self.shuffle,
      'num_workers': self.num_workers,
    }
  
  def reset(self):
    '''
    reset dataloader
    '''
    self.epoch = 0
    self.iter = 0
    if hasattr(self, '_iterator'):
      del self._iterator



