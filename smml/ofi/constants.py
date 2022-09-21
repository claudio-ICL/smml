from pathlib import Path


lobster_data_path: Path = Path('~/data/lobster')

message_cols: tuple[str, str, str, str, str, str] = (
    'time',
    'event_type',
    'order_id',
    'size',
    'price',
    'direction',
)

unique_time_event_cols: tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
] = (
    'nanoseconds',
    'milliseconds',
    'event_type',
    'size',
    'direction',
    'mid_price',
    'mid_price_delta',
    'smooth_mid',
    'bear_bull',
)

volume_samples_event_cols: tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
] = (
    'nanoseconds',
    'milliseconds',
    'event_type',
    'size',
    'direction',
    'mid_price',
    'smooth_mid',
    'bear_bull',
    'path_label',
    'tot_volume',
    'tot_bid_volume',
    'tot_ask_volume',
    'volume_imbalance',
)
