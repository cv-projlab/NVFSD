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

    def save(self, model:nn.Module, optimizer:torch.optim, scheduler:torch.optim.lr_scheduler, epoch, save_path):
        # Save last epoch model weights
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, save_path)

    def save_checkpoint(self, cur_loss:float, cur_acc:float, model:nn.Module, optimizer:torch.optim, scheduler:torch.optim.lr_scheduler, epoch, run_dir:Union[Path, str], model_name:Union[Path, str]) -> None:
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
        self.save_checkpoint(model, optimizer, scheduler, epoch, weights_dir/f"{model_name}.pth")
        
        # Save best model weights
        extra = self._extra_name_solver(cur_acc, cur_loss)
        if not extra: return # Not best epoch
        
        # Match files 
        matching_files=list(weights_dir.rglob(f'{model_name}{extra}.pth'))
        assert len(matching_files) <= 1, f'Can only have 1 or 0 matching files for {model_name}{extra}.pth'

        try: Path(matching_files[0]).unlink() # Remove existing matching file
        except IndexError: pass # No file to remove
            
        self.save_checkpoint(model, optimizer, scheduler, epoch, weights_dir/f"{model_name}{extra}.pth")
        
    def _extra_name_solver(self, cur_acc:float, cur_loss:float) -> str:
        acc_is_best = self.acc_tracker(cur_acc)
        loss_is_best = self.loss_tracker(cur_loss)
        
        if acc_is_best and not loss_is_best: return '_bestacc'
        if not acc_is_best and loss_is_best: return '_bestloss'
        if acc_is_best and loss_is_best: return '_bestaccloss'
        return '' # Not best epoch

    def load_checkpoint(self, checkpoint_path:str, model:nn.Module, optimizer:torch.optim, scheduler:torch.optim.lr_scheduler) -> int:
        """ Loads checkpoint model, optimizer and scheduler weights

        Args:
            checkpoint_path (str): Checkpoint path
            model (nn.Module): Model
            optimizer (torch.optim): _description_
            scheduler (torch.optim.lr_scheduler): _description_

        Returns:
            int: checkpoint epoch
        """
        if not os.path.isfile(checkpoint_path): return 0 # Starting epoch = 0

        # Load checkpoint to cpu
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model_state_dict=checkpoint['model_state_dict']
        optimizer_state_dict=checkpoint['optimizer_state_dict']
        scheduler_state_dict=checkpoint['scheduler_state_dict']
        epoch = checkpoint['epoch']

        # Deal with data parallel model.module state dict
        new_model_state_dict = OrderedDict()
        for k,v in model_state_dict.items(): 
            new_model_state_dict[k.replace('module.', '')] = v
        del model_state_dict

        # Update model state dict
        model.load_state_dict(new_model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

        return epoch

    def __repr__(self) -> str:
        return f'''Loss tracker -> {self.loss_tracker}
Accuracy tracker -> {self.acc_tracker}
'''        
