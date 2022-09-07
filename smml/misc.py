from typing import Optional
import numpy as np
from smml import misc
from itertools import permutations


def sequential_index(
        multiindex: list[int],
        dim: int = 1,
) -> int:
    assert dim > 0
    n: int = 0
    b: int = 1
    for i in reversed(multiindex):
        assert i > 0
        n += b*i
        b *= dim
    return n


def multiindex(
        sequential_index: int,
        dim: int = 1,
) -> list[int]:
    assert dim > 0
    n: int = sequential_index - 1
    alpha: list[int] = []
    while n >= 0:
        alpha.append(1 + n % dim)
        n = -1 + n // dim
    multiindex: list[int] = list(reversed(alpha))
    return multiindex


def monomial(
        X: np.ndarray,
        multiindex: list[int],
        time_index: int = -1,
) -> float:
    """
    `multiindex` is expected as a list of integers from 1 to d, where
    `d = X.shape[1]`
    """
    assert X.ndim == 2
    _mi = tuple([x-1 for x in multiindex])
    return np.prod(X[time_index, _mi])


def mononial_of_marginal_of_path(
    X: np.ndarray,
    alpha: Optional[list[int]] = None,
    MAX_LEVEL: int = 8,
) -> float:
    dim: int = X.shape[1]
    alpha = alpha or list(np.random.choice(np.arange(1, dim+1),
                                           size=1+np.random.choice(MAX_LEVEL)))
    mon_: float = monomial(X, alpha, time_index=-1)
    return mon_


def entry_of_symmetric_part(
    SX: np.ndarray,
    dim: int,
    alpha: Optional[list[int]] = None,
    MAX_LEVEL: int = 8,
) -> float:
    alpha = alpha or list(np.random.choice(np.arange(1, dim+1),
                                           size=1+np.random.choice(MAX_LEVEL)))
    assert len(SX) >= dim ** len(alpha)
    entry: float = .0
    for sigma_alpha in permutations(alpha):
        n: int = misc.sequential_index(list(sigma_alpha), dim)
        entry += SX[n]
    return entry
