from functools import partial
import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed, collate_fcn

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

# location generator (point, segment, etc)
generators = {}
def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator

def make_dataset(name, is_training, split, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers, num_classes, max_seq_len, **kwargs):
    """
        A simple dataloder builder
    """
    collate_function = partial(collate_fcn, training=is_training, num_classes=num_classes, max_seq_len=max_seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True,
        collate_fn=collate_function,
        prefetch_factor=20,
    )
    return loader

def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator