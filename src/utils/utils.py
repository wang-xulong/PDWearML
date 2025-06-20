import os
import random
from typing import Callable, Union
from omegaconf import DictConfig
from optuna import Trial
import numpy as np
from decimal import Decimal


def suggest_int(trial: Trial, cfg: DictConfig, *route: str) -> int:
    return _suggest(trial.suggest_int, cfg, *route)


def suggest_float(trial: Trial, cfg: DictConfig, *route: str, float_round: int) -> float:
    v = _suggest(trial.suggest_float, cfg, *route)
    # approx the params
    if float_round > 0:
        v = round(v, float_round)
    return v


def _suggest(func: Callable, cfg: DictConfig, *route: str) -> Union[float, int]:
    d = cfg
    for p in route:
        d = d[p]
    name = route[-1]
    if isinstance(d, DictConfig):
        low = d['low']
        high = d['high']
        if 'log' in d and d['log']:
            v = func(name, low, high, log=True)
        elif 'step' in d:
            step = d['step']
            v = func(name, low, high, step=step)
        else:
            v = func(name, low, high)
    else:
        v = d
    print(f'Fetching name: {name}={v} from {"/".join(route)}', flush=True)
    v = round(v, 5)
    return v


def set_seed(seed: int):
    print(f'Set seed={seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return seed


# Define __all__ to include all the functions you want to export
__all__ = [
    "suggest_int",
    "suggest_float",
    "_suggest",
    "set_seed"
]
