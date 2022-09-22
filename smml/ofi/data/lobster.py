from typing import Dict, Union, Callable, Optional
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

    def unique_time_ob(self,
                       t0: Optional[int] = None,
                       t1: Optional[int] = None,
                       alpha: float = .00035,
                       ) -> pd.DataFrame:
        t0 = t0 or self.df['time'].iloc[0]
        t1 = t1 or self.df['time'].iloc[-1]
        idx = (t0 <= self.df['time']) & (
            self.df['time'] <= t1)
        df: pd.DataFrame = pd.DataFrame(self.df.loc[idx])
        return unique_time_ob(df, alpha=alpha, levels=self.id.levels)


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


def volume_cols(levels: int) -> pd.Index:
    def ask_volume(n): return f'ask_volume_{n}'
    def bid_volume(n): return f'bid_volume_{n}'
    bid_cols: list[str] = [
        bid_volume(n) for n in range(levels, 0, -1)]
    ask_cols: list[str] = [
        ask_volume(n) for n in range(1, levels+1)]
    cols: list[str] = bid_cols + ask_cols
    return pd.Index(cols)


def bid_volume_cols(levels: int) -> pd.Index:
    def bid_volume(n): return f'bid_volume_{n}'
    bid_cols: list[str] = [
        bid_volume(n) for n in range(levels, 0, -1)]
    return pd.Index(bid_cols)


def ask_volume_cols(levels: int) -> pd.Index:
    def ask_volume(n): return f'ask_volume_{n}'
    ask_cols: list[str] = [
        ask_volume(n) for n in range(1, levels+1)]
    return pd.Index(ask_cols)


def message_cols() -> pd.Index:
    return pd.Index(list(constants.message_cols))


def load_orderbook(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = orderbook_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
        pd.read_csv(
            fp, header=None, index_col=None)
    )
    df.columns = _orderbook_cols(ldi.levels)
    df.insert(0, 'mid_price',
              (df['ask_price_1'] + df['bid_price_1']) // 2)
    df.insert(1, 'mid_price_delta',
              df['mid_price'].diff().fillna(0).astype(int))
    return df


def load_message(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = message_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
        pd.read_csv(
            fp, header=None, index_col=None,
        )
    )
    df.columns = message_cols()
    # time is expressed as integers representing nanosecond after market open
    df['time'] = ((df['time'] - df['time'].min())
                  * 1e9).fillna(-1).astype(np.int64)
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
            df['event_type'], df['direction']):
        assert len(str(event_type)) == len(str(direction))


def unique_time_ob(
        df: pd.DataFrame,
        alpha: float = .00035,
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
    st.rename(columns={'time': 'nanosecond'}, inplace=True)
    st['millisecond'] = np.ceil(
        st['nanosecond'] / 1000000).fillna(-1).astype(np.int64)
    st.sort_values(by='nanosecond', inplace=True)
    st['smooth_mid'] = st['mid_price'].ewm(
        alpha=alpha).mean()
    smid_delta = st['smooth_mid'].diff().fillna(0.)
    smid_delta_sign = pd.Series(np.sign(smid_delta)).fillna(0).astype(np.int64)
    assert len(smid_delta_sign.unique()) > 1
    st['bear_bull'] = smid_delta_sign.replace(
        0, np.nan).ffill(downcast='infer').bfill(  # type: ignore
        downcast='infer').astype(np.int64)
    st['segment_label'] = st['bear_bull'].diff().fillna(
        0).abs().div(2).cumsum().astype(np.int64)
    idx_bear = st['bear_bull'] == -1
    idx_bull = st['bear_bull'] == 1
    for seg in st.loc[idx_bull, 'segment_label'].unique():
        idx = st['segment_label'].isin([seg])
        idxmax = st.loc[idx, 'mid_price'].iloc[::-1].idxmax()
        idx_change = idx & (st.index >= idxmax)
        st.loc[idx_change, 'bear_bull'] = -1
    for seg in st.loc[idx_bear, 'segment_label'].unique():
        idx = st['segment_label'].isin([seg])
        idxmin = st.loc[idx, 'mid_price'].iloc[::-1].idxmin()
        idx_change = idx & (st.index >= idxmin)
        st.loc[idx_change, 'bear_bull'] = 1

    obcols: list[str] = list(_orderbook_cols(levels))
    sorted_cols: list[str] = list(constants.unique_time_event_cols) + obcols
    st = st.reindex(sorted_cols, axis=1)
    return st
