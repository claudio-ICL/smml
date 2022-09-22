from typing import Optional, Tuple, Callable
from functools import partial
import numpy as np
import pandas as pd
import esig
import torch
from torch import Tensor
from torch.utils.data import Dataset
from smml.ofi import constants
from smml.ofi.data import lobster
from smml.ofi.data.lobster import LobsterData, LobsterDataIdentifier


def _from_ob_to_volume_samples_dataframe(
        orderbook: pd.DataFrame,  # output of unique_time_ob
        levels: int = 3,
) -> pd.DataFrame:
    df: pd.DataFrame = orderbook.copy()
    df['tot_vol'] = df.loc[:, lobster.volume_cols(levels)].sum(axis=1)
    for col in lobster.volume_cols(levels):
        df[col] = df[col].astype(np.float64).div(df['tot_vol'])
    bid_cols = lobster.bid_volume_cols((levels))
    ask_cols = lobster.ask_volume_cols((levels))
    df['tot_bid_volume'] = df[bid_cols].sum(axis=1)
    df['tot_ask_volume'] = df[ask_cols].sum(axis=1)
    df['volume_imbalance'] = df['tot_bid_volume'] - df['tot_ask_volume']
    df['path_label'] = df['bear_bull'].diff().fillna(
        0).abs().div(2).cumsum().astype(np.int64)
    sorted_cols: list[str] = list(
        constants.volume_samples_event_cols) + list(lobster.volume_cols(levels))
    df = df.reindex(sorted_cols, axis=1)
    return df


def from_ob_to_volume_samples(
        orderbook: pd.DataFrame,  # output of unique_time_ob
        bear_bull: int = 1,
        levels: int = 3,
        include_volume_imbalance: bool = True,
) -> list[np.ndarray]:
    df: pd.DataFrame = _from_ob_to_volume_samples_dataframe(
        orderbook=orderbook,
        levels=levels,
    )
    data_cols: list[str]
    if include_volume_imbalance:
        data_cols = ['volume_imbalance'] + list(lobster.volume_cols(levels))
    else:
        data_cols = list(lobster.volume_cols(levels))
    cols: list[str] = ['path_label'] + data_cols
    idx_bear_bull = df['bear_bull'].isin([bear_bull])
    assert idx_bear_bull.sum() > 0, f'No instances found'
    samples: pd.DataFrame = pd.DataFrame(df.loc[idx_bear_bull, cols].copy())
    paths: list[np.ndarray] = []
    for pl in samples['path_label'].unique():
        idx = samples['path_label'].isin([pl])
        path: np.ndarray = np.array(
            samples.loc[idx, data_cols],
            dtype=np.float64,
        )
        paths.append(path)
    return paths


def from_volume_samples_to_sig_samples(
        paths: list[np.ndarray],
        sigdegree: int,
) -> Tensor:
    logsignatures: list[np.ndarray] = [
        np.expand_dims(
            esig.tosig.stream2logsig(path, sigdegree),  # type: ignore
            axis=0,
        )
        for path in paths
    ]
    logsigs: np.ndarray = np.concatenate(logsignatures, axis=0)
    sample_tensor: Tensor = torch.from_numpy(logsigs)
    return sample_tensor


def signature_samples(
        ldis: list[LobsterDataIdentifier],
        t0: Optional[int] = None,  # in nanoseconds
        t1: Optional[int] = None,  # in nanoseconds
        ewm_decay: float = .00035,
        bear_bull: int = 1,  # 1 for bull, -1 for bear
        book_levels: int = 3,  # limit order book levels
        include_volume_imbalance: bool = True,
        sigdegree: int = 4,
) -> Tensor:
    orderbooks: list[pd.DataFrame] = []
    for ldi in ldis:
        ld: LobsterData = LobsterData(ldi)
        df: pd.DataFrame = ld.unique_time_ob(
            t0=t0,
            t1=t1,
            alpha=ewm_decay,
        )
        orderbooks.append(df)
    orderbook: pd.DataFrame = pd.concat(orderbooks, axis=0)
    paths: list[np.ndarray] = from_ob_to_volume_samples(
        orderbook=orderbook,
        bear_bull=bear_bull,
        levels=book_levels,
        include_volume_imbalance=include_volume_imbalance,
    )
    logsigs: Tensor = from_volume_samples_to_sig_samples(
        paths, sigdegree)
    return logsigs


def signature_samples_for_divergence(
        ldis: list[LobsterDataIdentifier],
        t0: Optional[int] = None,  # in nanoseconds
        t1: Optional[int] = None,  # in nanoseconds
        ewm_decay: float = .00035,
        bear_bull: Optional[int] = 1,  # 1 for bull, -1 for bear, None for both
        book_levels: int = 3,  # limit order book levels
        include_volume_imbalance: bool = True,
        sigdegree: int = 4,
) -> Tensor:
    orderbooks: list[pd.DataFrame] = []
    for ldi in ldis:
        ld: LobsterData = LobsterData(ldi)
        df: pd.DataFrame = ld.unique_time_ob(
            t0=t0,
            t1=t1,
            alpha=ewm_decay,
        )
        orderbooks.append(df)
    orderbook: pd.DataFrame = pd.concat(orderbooks, axis=0)

    def _logsig_tensor(bear_bull):
        paths: list[np.ndarray] = from_ob_to_volume_samples(
            orderbook=orderbook,
            bear_bull=bear_bull,
            levels=book_levels,
            include_volume_imbalance=include_volume_imbalance,
        )
        _logsigs: Tensor = from_volume_samples_to_sig_samples(
            paths, sigdegree)
        logsigs: Tensor = _logsigs[torch.randperm(len(_logsigs)), :]
        return logsigs
    logsigs: Tensor
    if bear_bull is None:
        bear_logsigs: Tensor = _logsig_tensor(-1)
        bull_logsigs: Tensor = _logsig_tensor(1)
        num_samples: int = min(bear_logsigs.size(0), bull_logsigs.size(0))
        assert num_samples > 0
        assert bear_logsigs.size(1) == bull_logsigs.size(1)
        logsigs = torch.cat((
            bear_logsigs[:num_samples, :],
            bull_logsigs[:num_samples, :],
        ), dim=1,
        )
    else:
        assert bear_bull in [-1,
                             1], f'bear_bull must be either -1, or 1, or None'
        _logsigs: Tensor = _logsig_tensor(bear_bull)
        cutpoint: int = _logsigs.size(0) // 2
        x: Tensor = _logsigs[: cutpoint, :]
        y: Tensor = _logsigs[cutpoint: 2*cutpoint, :]
        assert x.size(0) == y.size(0)
        logsigs = torch.cat((x, y), dim=1)

    return logsigs


class DivergenceSample(Dataset):
    def __init__(self, samples: Tensor, dim: int, sigdegree: int):
        self.samples: Tensor = samples
        self.number_of_samples = samples.size(0)
        self.dim: int = dim
        self.sigdegree: int = sigdegree
        logsigdim: int = esig.tosig.logsigdim(dim, sigdegree)  # type: ignore
        assert samples.size(1) == 2 * logsigdim
        self.logsigdim: int = logsigdim

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        i: int = idx % self.number_of_samples
        d: int = self.logsigdim
        sample: Tensor = self.samples[i, :]
        x: Tensor = sample[:d]
        y: Tensor = sample[d:]
        return x, y


def _signature_dataset(
        ldis: list[LobsterDataIdentifier],
        t0: Optional[int] = None,  # in nanoseconds
        t1: Optional[int] = None,  # in nanoseconds
        ewm_decay: float = .00035,
        bear_bull: Optional[int] = 1,  # 1 for bull, -1 for bear, None for both
        book_levels: int = 3,  # limit order book levels
        include_volume_imbalance: bool = True,
        sigdegree: int = 4,
) -> DivergenceSample:
    logsigs: Tensor = signature_samples_for_divergence(
        ldis=ldis,
        t0=t0,
        t1=t1,
        ewm_decay=ewm_decay,
        bear_bull=bear_bull,
        book_levels=book_levels,
        include_volume_imbalance=include_volume_imbalance,
        sigdegree=sigdegree,
    )
    dim: int = 2*book_levels + include_volume_imbalance
    dataset: DivergenceSample = DivergenceSample(
        samples=logsigs,
        dim=dim,
        sigdegree=sigdegree,
    )
    return dataset


bear_only_signature_dataset: Callable[..., DivergenceSample] = partial(
    _signature_dataset, bear_bull=-1,
)

bull_only_signature_dataset: Callable[..., DivergenceSample] = partial(
    _signature_dataset, bear_bull=1,
)

bear_bull_signature_dataset: Callable[..., DivergenceSample] = partial(
    _signature_dataset, bear_bull=None,
)
