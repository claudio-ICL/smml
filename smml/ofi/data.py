from typing import Dict, Union, Callable
from pathlib import Path
import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from smml.ofi import constants


class LobsterFileType(Enum):
    ORDERBOOK = 0
    MESSAGE = 1

    def __str__(self) -> str:
        return str(self.name)


@dataclass
class LobsterDataIdentifier:
    ticker: str
    date: datetime.date
    levels: int
    file_type: LobsterFileType
    t0: int = 34200000
    t1: int = 57600000

    @staticmethod
    def from_(other):
        ldi: LobsterDataIdentifier = LobsterDataIdentifier(
            ticker=other.ticker,
            date=other.date,
            levels=other.levels,
            file_type=other.file_type,
            t0=other.t0,
            t1=other.t1,
        )
        return ldi


LDI_EG: LobsterDataIdentifier = LobsterDataIdentifier(
    ticker='INTC',
    date=datetime.date(2012, 6, 21),
    levels=5,
    file_type=LobsterFileType.ORDERBOOK,
    t0=34200000,
    t1=57600000,
)


class LobsterData:
    def __init__(self, ldi: LobsterDataIdentifier):
        self.id: LobsterDataIdentifier = ldi
        self.df: pd.DataFrame = load_orderbook_and_message(ldi)

    def unique_time_ob(self, rolling_window: int) -> pd.DataFrame:
        return unique_time_ob(self.df, rolling_window, levels=self.id.levels)


def _file_path(ldi: LobsterDataIdentifier) -> Path:
    TICKER: str = ldi.ticker.upper()
    isodate: str = ldi.date.isoformat()
    ft: str = str(ldi.file_type).lower()
    file_stem: str = f'{TICKER}_{isodate}_{ldi.t0}_{ldi.t1}_{ft}_{ldi.levels}'
    fp: Path = constants.lobster_data_path / f'{file_stem}.csv'
    return fp


def orderbook_file_path(ldi: LobsterDataIdentifier) -> Path:
    ob_ldi: LobsterDataIdentifier = LobsterDataIdentifier.from_(ldi)
    ob_ldi.file_type = LobsterFileType.ORDERBOOK
    return _file_path(ob_ldi)


def message_file_path(ldi: LobsterDataIdentifier) -> Path:
    msg_ldi: LobsterDataIdentifier = LobsterDataIdentifier.from_(ldi)
    msg_ldi.file_type = LobsterFileType.MESSAGE
    return _file_path(msg_ldi)


def _orderbook_cols(levels: int) -> pd.Index:
    def ask_price(n): return [f'ask_price_{n}']
    def bid_price(n): return [f'bid_price_{n}']
    def ask_volume(n): return [f'ask_volume_{n}']
    def bid_volume(n): return [f'bid_volume_{n}']

    cols: list[str] = sum([
        sum([
            ask_price(n),
            ask_volume(n),
            bid_price(n),
            bid_volume(n),
        ], start=[])
        for n in range(1, 1+levels)
    ], start=[])
    return pd.Index(cols)


def _volume_cols(levels: int) -> pd.Index:
    def ask_volume(n): return f'ask_volume_{n}'
    def bid_volume(n): return f'bid_volume_{n}'
    bid_cols: list[str] = [
        bid_volume(n) for n in range(levels, 0, -1)]
    ask_cols: list[str] = [
        ask_volume(n) for n in range(1, levels+1)]
    cols: list[str] = bid_cols + ask_cols
    return pd.Index(cols)


def _message_cols() -> pd.Index:
    return pd.Index(list(constants.message_cols))


def load_orderbook(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = orderbook_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
        pd.read_csv(
            fp, header=None, index_col=None)  # type: ignore
    )
    df.columns = _orderbook_cols(ldi.levels)
    df.insert(0, 'mid_price',
              (df['ask_price_1'] + df['bid_price_1']) // 2)  # type: ignore
    df.insert(1, 'mid_price_delta',
              df['mid_price'].diff().fillna(0).astype(int))  # type:ignore
    return df


def load_message(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = message_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
        pd.read_csv(
            fp, header=None, index_col=None,  # type: ignore
        )
    )
    df.columns = _message_cols()
    # time is expressed as integers representing nanoseconds after market open
    df['time'] = ((df['time'] - df['time'].min())  # type: ignore
                  * 1e7).fillna(-1).astype(np.int64)  # type: ignore
    df.set_index(['time'], inplace=True)
    assert df.index.is_monotonic_increasing
    df.reset_index(inplace=True)
    return df


def test_mid_price_after_execution(df: pd.DataFrame):
    def one_side(direction: int):
        idx = (df['event_type'] == 4) & (df['direction'] == direction)
        assert np.all(direction * df.loc[idx, 'mid_price_delta'] <= 0)
    one_side(1)
    one_side(-1)


def load_orderbook_and_message(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    ob: pd.DataFrame = load_orderbook(ldi)
    msg: pd.DataFrame = load_message(ldi)
    assert len(ob) == len(msg)
    df: pd.DataFrame = pd.concat([msg, ob], axis=1)
    test_mid_price_after_execution(df)
    return df


def test_simultaneous_events(df: pd.DataFrame):
    for event_type, direction in zip(
            df['event_type'], df['direction']):    # type: ignore
        assert len(str(event_type)) == len(str(direction))


def unique_time_ob(
        df: pd.DataFrame,
        rolling_window: int,
        levels: int = 10,
) -> pd.DataFrame:
    agg: Dict[str, Union[Callable[..., int], 'str']] = {
        **dict(
            event_type=lambda xs: int(''.join([str(x) for x in xs])),
            size='sum',
            direction=lambda xs: int(
                ''.join(['1' if x == 1 else '2' for x in xs])),
            mid_price='last',
            mid_price_delta='sum',
        ),
        **{col: 'last' for col in _orderbook_cols(levels)},
    }
    st: pd.DataFrame = pd.DataFrame(
        df.groupby('time').agg(agg)
    )
    st.reset_index(inplace=True)
    test_simultaneous_events(st)
    st.rename(columns={'time': 'nanoseconds'}, inplace=True)
    st['milliseconds'] = np.ceil(
        st['nanoseconds'] / 1000).fillna(-1).astype(np.int64)  # type: ignore
    st.sort_values(by='nanoseconds', inplace=True)
    st['smooth_mid'] = st['mid_price'].rolling(  # type: ignore
        rolling_window).mean()  # type: ignore
    smid_delta = st['smooth_mid'].diff().fillna(0.)  # type: ignore
    smid_delta_sign = pd.Series(np.sign(smid_delta)).fillna(0).astype(np.int64)
    st['bear_bull'] = smid_delta_sign.replace(
        0, np.nan).ffill(downcast='infer').bfill(downcast='infer').astype(np.int64)  # type: ignore
    obcols: list[str] = list(_orderbook_cols(levels))
    sorted_cols: list[str] = list(constants.equispaced_event_cols) + obcols
    st = st.reindex(sorted_cols, axis=1)
    return st


def from_ob_to_volume_samples(
        orderbook: pd.DataFrame,  # output of unique_time_ob
        bear_bull: int = 1,
        levels: int = 3,
        include_spread: bool = True,
) -> list[np.ndarray]:
    df: pd.DataFrame = orderbook.copy()
    idx_bear_bull = df['bear_bull'].isin([bear_bull])
    assert idx_bear_bull.sum() > 0, f'No instances found'
    df['spread'] = (df['ask_price_1'] - df['bid_price_1']) / \
        100  # expressed in ticks
    df['tot_vol'] = df.loc[:, _volume_cols(levels)].sum(axis=1)
    for col in _volume_cols(levels):
        df[col] = df[col].astype(np.float64).div(df['tot_vol'])
    df['path_label'] = df['bear_bull'].diff().fillna(
        0).abs().div(2).cumsum().astype(np.int64)
    data_cols: list[str]
    if include_spread:
        data_cols = ['spread'] + list(_volume_cols(levels))
    else:
        data_cols = list(_volume_cols(levels))
    cols: list[str] = ['path_label'] + data_cols
    samples = pd.DataFrame(df.loc[idx_bear_bull, cols].copy())
    paths: list[np.ndarray] = []
    for pl in samples['path_label'].unique():
        idx = samples['path_label'].isin([pl])
        path: np.ndarray = np.expand_dims(
            np.array(
                samples.loc[idx, data_cols],
                dtype=np.float64,
            ),
            axis=0,
        )
        paths.append(path)
    return paths
