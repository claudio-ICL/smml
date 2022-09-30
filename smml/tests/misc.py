from esig import tosig as ts
from smml import misc
import numpy as np


def index_conversion(
    MAX_DIM: int = 6,
    MAX_LEVEL: int = 8,
):
    dim: int = 2 + np.random.choice(MAX_DIM - 1)
    n: int
    alpha: list[int]
    # right inverse
    n = 1 + np.random.choice(MAX_LEVEL * MAX_DIM)
    alpha = misc.multiindex(n, dim)
    assert misc.sequential_index(alpha, dim) == n
    # left inverse
    alpha = list(np.random.choice(np.arange(1, dim+1),
                 size=1+np.random.choice(MAX_LEVEL)))
    n = misc.sequential_index(alpha, dim)
    assert misc.multiindex(n, dim) == alpha
    # consistency with ts.sigkeys
    alpha = list(np.random.choice(np.arange(1, dim+1),
                 size=2+np.random.choice(4)))
    n = misc.sequential_index(alpha, dim)
    siglevel: int = max(2, len(alpha))
    try:
        sigkeys: list[str] = ts.sigkeys(dim, siglevel).split(' ')[
            1:]  # type: ignore
    except (RuntimeError, SystemError) as e:
        print(e)
        print(f'dim = {dim}')
        print(f'siglevel = {siglevel}')
        raise (e)
    assert eval(sigkeys[n]) == tuple(alpha)
