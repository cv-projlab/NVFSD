from collections import OrderedDict

import torch
import torch.nn as nn

from ..env import *


class FloatTracker:
    def __init__(self, track_highest:bool=True, lowest:float=0., highest:float=1.) -> None:
        assert lowest < highest, f'Expected lowest < highest. Got {lowest} < {highest}'

        self.lowest=lowest
        self.highest=highest
        self.track_highest=track_highest
        self.best = lowest if track_highest else highest

    def __call__(self, value:float) -> bool:
        assert self.lowest <= value <= self.highest, f'Value must be in range {self.lowest} <= value <= {self.highest}. Got value = {value}'
        
        f = max if self.track_highest else min
        self.best = f(self.best, value) 

        is_best = self.best == value
        return is_best
 
    def __repr__(self) -> str:
        return f'''Tracking {"highest" if self.track_highest else "lowest"} value
LOWEST <= BEST <= HIGHEST
{self.lowest} <= {self.best} <= {self.highest}
'''


class Weights:
    def __init__(self) -> None:
        self.loss_tracker = FloatTracker(track_highest=False, highest=1e5)
        self.acc_tracker = FloatTracker(track_highest=True)

    def save(self, cur_loss:float, cur_acc:float, model:nn.Module, run_dir:Union[Path, str], model_name:Union[Path, str]) -> None:
        """ Saves a model state dict

        Args:
            cur_loss (float): Current loss value
            cur_acc (float): Current accuracy value
            model (nn.Module): Model
            run_dir (Union[Path, str]): Root run directory
            model_name (Union[Path, str]): Model Name

        """
        if isinstance(run_dir, str): run_dir = Path(run_dir)
        weights_dir = run_dir / model_name

        # Save last epoch model weights
        torch.save(model.state_dict(), weights_dir/f"{model_name}.pth")
        
        # Save best model weights
        extra = self._extra_name_solver(cur_acc, cur_loss)
        if not extra: return # Not best epoch
        
        # Match files 
        matching_files=list(weights_dir.rglob(f'{model_name}{extra}.pth'))
        assert len(matching_files) <= 1, f'Can only have 1 or 0 matching files for {model_name}{extra}.pth'

        try: Path(matching_files[0]).unlink() # Remove existing matching file
        except IndexError: pass # No file to remove
            
        torch.save(model.state_dict(), weights_dir/f'{model_name}{extra}.pth')
        
    def _extra_name_solver(self, cur_acc:float, cur_loss:float) -> str:
        acc_is_best = self.acc_tracker(cur_acc)
        loss_is_best = self.loss_tracker(cur_loss)
        
        if acc_is_best and not loss_is_best: return '_bestacc'
        if not acc_is_best and loss_is_best: return '_bestloss'
        if acc_is_best and loss_is_best: return '_bestaccloss'
        return '' # Not best epoch

        
    def load(self, model_path:str, model:nn.Module):
        """ Loads model weights

        Args:
            model_path (str): Weights path
            model (nn.Module): Model
        """
        # Load state dict to cpu
        state_dict = torch.load(model_path, map_location='cpu')

        # Deal with data parallel model.module state dict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items(): 
            new_state_dict[k.replace('module.', '')] = v
        del state_dict

        # Update model state dict
        model.load_state_dict(new_state_dict, strict=True)

    def __repr__(self) -> str:
        return f'''Loss tracker -> {self.loss_tracker}
Accuracy tracker -> {self.acc_tracker}
'''        
