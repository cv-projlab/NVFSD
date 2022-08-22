from ..env import *
from ..imop.transformations import affine as affineT
from ..imop.transformations import random_erase_box as random_erase_boxT


class rnd_erase:
    def __init__(self, prob:float=0.5) -> None:
        self.defaults = {'prob':prob}

    def __call__(self, clip:np.ndarray) -> np.ndarray:
        return random_erase_boxT(clip, **self.defaults)


class affine:
    def __init__(self, angle:int=15, scale:float=0.75, deltax:int=30, deltay:int=30) -> None:
        self.defaults = {'angle':angle, 'scale':scale, 'deltax':deltax, 'deltay':deltay}

    def __call__(self, clip:np.ndarray) -> np.ndarray:
        return affineT(clip, **self.defaults)


class normalize:
    def __init__(self) -> None:
        pass
    
    def __call__(self, clip:np.ndarray) -> np.ndarray:
        return  (clip / 255.).astype(FLOAT32)
