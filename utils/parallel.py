import os
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Callable


class Parallel:
    def __init__(self, nprocess=mp.cpu_count()):
        self.nprocess = nprocess
        self.pool = mp.Pool(processes=nprocess)
        self.out = {}

    def add(self, name: str, func: Callable, args: List, show=True):
        if show:
            self.out[name] = list(
                self.pool.starmap(func, tqdm(args, ncols=80, total=len(args)))
            )
        else:
            self.out[name] = list(self.pool.starmap(func, args))

        return self.out[name]

    def join(self):
        self.pool.close()
        self.pool.join()

    @property
    def results(self):
        return self.out
