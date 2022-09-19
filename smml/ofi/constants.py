from pathlib import Path


lobster_data_path : Path = Path('~/data/lobster')

message_cols: tuple[str, str, str, str, str, str] = (
        'time',
        'event_type',
        'order_id',
        'size',
        'price',
        'direction',
        )

