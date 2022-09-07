import numpy as np


def sample(
        t0: float = .0,
        t1: float = 1.,
        dt: float = 1e-6,
        dim: int = 1,
) -> np.ndarray:
    assert dt > 0
    N: int = int((t1 - t0) / dt)
    assert N > 0
    scale: float = np.sqrt(dt)
    dX: np.ndarray = np.random.normal(
        loc=.0,
        scale=scale,
        size=(N, dim),
    )
    X0: np.ndarray = np.zeros((1, dim), dtype=float)
    X: np.ndarray = np.concatenate((X0, np.cumsum(dX, axis=0)), axis=0)
    return X
