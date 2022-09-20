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

equispaced_event_cols: tuple[
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
