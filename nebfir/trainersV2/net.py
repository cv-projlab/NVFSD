from enum import Enum

import torch

from ..model.net_enums import criterions, models, optimizers, schedulers
from ..tools.tools_basic import dict2str
from .trainer_params import TrainerParams


def get_(key: str, enum_: Enum, *args, **kwargs):
    assert key in list(enum_.__members__.keys()), f"Got an unknown {enum_.__name__} key: < {key} > . Choose from {list(enum_.__members__.keys())}"

    return enum_[key].value(*args, **kwargs)


class Net:
    def __init__(self, trainer_params: TrainerParams):
        self.trainer_params = trainer_params
        self.using_data_parallel = self.trainer_params.data_parallel
        
        self.reset()
        
    def reset(self):
        self.device = self.trainer_params.device
        self.model     =  get_(self.trainer_params.architecture_key,  models, **self.trainer_params.architecture_args_dict)
        self.criterion =  get_(self.trainer_params.criterion_key, criterions, **self.trainer_params.criterion_args_dict)
        self.optimizer =  get_(self.trainer_params.optimizer_key, optimizers, self.model.parameters(), **self.trainer_params.optimizer_args_dict)
        self.scheduler =  get_(self.trainer_params.scheduler_key, schedulers, self.optimizer, **self.trainer_params.scheduler_args_dict)

        self.keys = {
            'architecture': self.trainer_params.architecture_key, 
            'optimizer': self.trainer_params.optimizer_key, 
            'scheduler': self.trainer_params.scheduler_key, 
            'criterion': self.trainer_params.criterion_key,
            }

        self.to_device()
        
    def to_device(self, device:str=None):
        """ Send model and criterion to device

        Args:
            device (str, optional): Device. Defaults to None.
        """
        if self.using_data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.criterion = torch.nn.DataParallel(self.criterion)
            return

        self.model.to(device or self.device)
        self.criterion.to(device or self.device)

        
    def __repr__(self) -> str:
        return f"""
Built Net:
{dict2str(self.keys, level=1, indentation=2)}
  device: {self.device if not self.using_data_parallel else 'cuda:'+'|'.join(map(str, self.model.device_ids))}
"""



#ENDFILE
