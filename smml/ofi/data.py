from typing import Dict
from pathlib import Path
import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from smml.ofi import constants

class LobsterFileType(Enum):
    ORDERBOOK=0
    MESSAGE=1
    def __str__(self)-> str:
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
                levels=10,
                file_type=LobsterFileType.ORDERBOOK,
                t0=34200000,
                t1=57600000,
                )

class LobsterData:
    def __init__(self, ldi: LobsterDataIdentifier):
        self.id: LobsterDataIdentifier = ldi
        self.df: pd.DataFrame = load_orderbook_and_message(ldi)

    def equispaced_orderbook(self, interval_len: np.int64) -> pd.DataFrame:
        return equispaced_ob(self.df, interval_len)


def _file_path(ldi: LobsterDataIdentifier) -> Path:
    TICKER: str = ldi.ticker.upper()
    isodate: str = ldi.date.isoformat()
    ft: str= str(ldi.file_type).lower()
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
    ask_price= lambda n: [ f'ask_price_{n}']
    bid_price= lambda n: [ f'bid_price_{n}']
    ask_volume= lambda n: [ f'ask_volume_{n}']
    bid_volume= lambda n: [ f'bid_volume_{n}']

    cols: list[str] = sum([
        sum([
                ask_price(n),
                ask_volume(n),
                bid_price(n),
                bid_volume(n),
        ], start=[])
        for n in range (1, 1+levels)
        ], start=[])
    return pd.Index(cols)

def _message_cols() -> pd.Index:
    return pd.Index(list(constants.message_cols))

def load_orderbook(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = orderbook_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
            pd.read_csv(
            fp, header=None, index_col=None)
            )
    df.columns = _orderbook_cols(ldi.levels)
    df.insert(0, 'mid_price',  (df['ask_price_1'] + df['bid_price_1']) // 2)
    df.insert(1, 'mid_price_delta', df['mid_price'].diff().fillna(0).astype(int))
    return df

def load_message(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    fp: Path = message_file_path(ldi)
    df: pd.DataFrame = pd.DataFrame(
            pd.read_csv(
            fp, header=None, index_col=None,
            )
            )
    df.columns = _message_cols()
    # time is expressed as integers representing nanoseconds after market open
    df['time'] = ((df['time']  - df['time'].min()) * 1e7).fillna(-1).astype(np.int64)
    df.set_index(['time'], inplace=True)
    assert df.index.is_monotonic_increasing
    df.reset_index(inplace=True)
    return df

def test_mid_price_after_execution(df: pd.DataFrame):
    def one_side(direction: int):
        import pdb
        pdb.set_trace()
        idx = (df['event_type'] == 4) & (df['direction']==direction)
        assert np.all(direction * df.loc[idx, 'mid_price_delta'] <=0)
    one_side(1)
    one_side(-1)


def load_orderbook_and_message(ldi: LobsterDataIdentifier) -> pd.DataFrame:
    ob: pd.DataFrame = load_orderbook(ldi)
    msg: pd.DataFrame = load_message(ldi)
    assert len(ob) == len(msg)
    df: pd.DataFrame = pd.concat([msg, ob], axis=1)
    test_mid_price_after_execution(df)
    return df

def equispaced_ob(
        df: pd.DataFrame, 
        interval_len: np.int64, # in milliseconds
) -> pd.DataFrame:
    st: pd.DataFrame = pd.DataFrame(
            df.groupby('time').last().reset_index()
            )
    st.rename(columns = {'time': 'nanoseconds'}, inplace=True)
    st['milliseconds'] = np.ceil(st['nanoseconds'] * 1000).fillna(-1).astype(np.int64)
    st['delta_t'] = np.ceil(
            (st['milliseconds'] - st['milliseconds'].min()) / interval_len
    ).astype(np.int64)
    delta_t = pd.DataFrame({'delta_t': np.arange(
        1 + np.ceil(
            (st['milliseconds'].max() - st['milliseconds'].min()) / interval_len
    ))})
    st = pd.DataFrame(
            st.merge(delta_t, on='delta_t', how='outer', validate='m:1').ffill()
            )
    st['smooth_mid'] = st['mid_price'].rolling(interval_len).mean()
    return st





