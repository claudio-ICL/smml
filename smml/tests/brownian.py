from esig import tosig as ts
from smml.examples import brownian
from typing import Optional
from smml import misc
import numpy as np


def symmetric_part(
    dim: Optional[int] = None,
    MAX_DIM: int = 8,
    MAX_LEVEL: int = 8,
    atol=1e-9,
    rtol=1e-12,
):
    dim = dim or 2 + np.random.choice(MAX_DIM - 1)
    alpha: list[int] = list(
        np.random.choice(
            np.arange(1, dim+1),
            size=1+np.random.choice(MAX_LEVEL)
        )
    )
    X: np.ndarray = brownian.src.sample(
        dt=1e-5,
        dim=dim,
    )
    siglevel: int = max(2, len(alpha))
    try:
        SX: np.ndarray = ts.stream2sig(X, siglevel)  # NOQA
    except (RuntimeError, SystemError) as e:
        print('Could not compute signature')
        print(e)
        print(f'dim = {dim}')
        print(f'siglevel = {siglevel}')
        raise (e)
    monomial: float = misc.mononial_of_marginal_of_path(X, alpha, MAX_LEVEL)
    entry_of_symmetric_part: float = misc.entry_of_symmetric_part(
        SX, dim, alpha, MAX_LEVEL)
    assert np.isclose(monomial, entry_of_symmetric_part, atol=atol, rtol=rtol)
