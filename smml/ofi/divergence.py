from typing import Optional, Iterator
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from smml.ofi import donsker_varadhan as dv
from smml.ofi.data.samples import DivergenceSample


class Divergence:
    def __init__(self,
                 dataset: DivergenceSample,
                 test_function: Optional[dv.TestFunction] = None,
                 dtype=torch.float64,
                 empirical_sample_size: int = 1,
                 ):
        self.dataset: Dataset = dataset
        self.test_function: dv.TestFunction = (
            test_function or dv.TestFunction(
                dataset.logsigdim,
                dtype=dtype,
            )
        )
        assert empirical_sample_size > 0
        self.sampler: DataLoader = DataLoader(
            self.dataset,
            batch_size=empirical_sample_size,
            sampler=RandomSampler(
                self.dataset,  # type: ignore
                replacement=True,
            ),
        )
        self.sample: Iterator = iter(self.sampler)

    def train(self,
              optimizer=None,
              learning_rate: float = 1e-3,
              batch_size: int = 1000,
              shuffle: bool = True,
              num_of_epochs: int = 1,
              verbose: bool = True,
              ):
        for epoch in range(num_of_epochs):
            if verbose:
                print(f'\n\nEPOCH: {epoch}\n')
            dl: DataLoader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            f_star: dv.TestFunction = dv.train(
                dataloader=dl,
                f=self.test_function,
                optimizer=optimizer,
                learning_rate=learning_rate,
                verbose=verbose,
            )
            self.test_function = f_star

    def __call__(self,
                 x: Optional[Tensor] = None,
                 y: Optional[Tensor] = None,
                 ):
        try:
            x_, y_ = next(self.sample)
        except StopIteration:
            self.sample = iter(self.sampler)
            x_, y_ = next(self.sample)
        samples_from_p: Tensor
        samples_from_q: Tensor
        if x is None:
            samples_from_p = x_
        else:
            samples_from_p = x
        if y is None:
            samples_from_q = y_
        else:
            samples_from_q = y
        f: dv.TestFunction = self.test_function
        v: Tensor = dv.V(samples_from_p, samples_from_q, f)
        return v.item()
