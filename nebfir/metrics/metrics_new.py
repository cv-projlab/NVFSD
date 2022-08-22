from typing import Dict, List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Metrics:
    " Metrics base class for loss and accuracy tracking "
    def __init__(self, DEBUG:bool=False, *args, **kwargs) -> None:
        self.DEBUG = DEBUG
        self.reset()
            
    def reset(self):
        self.iteration_value: float = 0.0
        self.iteration_value_list: List = []
        self.mean_value: float = 0.0
        
    def __call__(self, p: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor], convert_target2onehot:bool=False):
        return self.update(p=p, t=t, convert_target2onehot=convert_target2onehot)

    def update(self, p: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor], convert_target2onehot=False):
        """ Updates Metrics

        Args:
            p (Union[np.ndarray, torch.Tensor]): Predictions
            t (Union[np.ndarray, torch.Tensor]): Targets
            convert_target2onehot (bool, optional): Target is in form of One-Hot Encoding. Defaults to False.
            
        TODO: assert input types and shapes
        """
        # if 'loss' in self.__class__.__name__.lower():
        #     assert isinstance(p, (torch.Tensor)) and isinstance(t, (torch.Tensor)), f'Prediction p:{type(p)} and Target t:{type(t)} must have the same type and be of type torch.Tensor' 
        # else:      
        #     assert isinstance(p, (np.ndarray,torch.Tensor)) and isinstance(t, (np.ndarray,torch.Tensor)), f'Prediction p:{type(p)} and Target t:{type(t)} must have the same type and be of either np.ndarray or torch.Tensor' 
        
        # if convert_target2onehot:
        #     assert p.shape == t.shape , f'Prediction p:{p.shape} and Target t:{t.shape} must have the same shape'
        # else:
        #     assert (p.shape[0] == t.shape[0]) and t.ndim == 1 , f'Prediction p:{p.shape} and Target t:{t.shape} must have the same length and Target must have only 1 dimension'       
        
        if self.DEBUG:
            print('\nconverttarget2_on_hot: ', convert_target2onehot)
            print('Prediction type: ', type(p))
            print('Target type: ', type(t))
            print('Prediction: ', p)
            print('Target: ', t)
            
    def toggle_debug(self):
        self.DEBUG = not self.DEBUG
        
    def __repr__(self) -> str:
        str_ = f"Metrics: {self.__class__.__name__}\n"
        str_+= f"Criterion: {self.criterion.__class__.__name__}\n" if 'loss' in self.__class__.__name__.lower() else ""
        str_+= f"Debug: {self.DEBUG}"
        return str_

    
class LossMetrics(Metrics):
    """ Loss Metrics based on the Metrics Class """
    def __init__(self, criterion, DEBUG: bool = False, *args, **kwargs) -> None:
        """ Initializes the loss metrics

        Args:
            criterion : The loss criterion to update the model
        """
        self.criterion = criterion

        super().__init__(DEBUG=DEBUG, *args, **kwargs)
        

    
    def update(self, p: torch.Tensor, t: torch.Tensor, convert_target2onehot=False) -> torch.Tensor:
        """ Updates Loss

        Args:
            p (Union[np.ndarray, torch.Tensor]): Predictions
            t (Union[np.ndarray, torch.Tensor]): Targets
            convert_target2onehot (bool, optional): Target is in form of One-Hot Encoding. Defaults to False.

        Returns:
            torch.Tensor: Loss
        """
        super().update(p, t, convert_target2onehot=convert_target2onehot)
        
        if convert_target2onehot:
            t = F.one_hot(t, p.shape[1])
                    
        if self.DEBUG:
            print('Target:', t)
            
        self.iteration_value = self.criterion_(p, t)
        
        self.iteration_value_list.append(self.iteration_value.item())
        self.mean_value = np.array(self.iteration_value_list).mean(dtype=float)
        
        if self.DEBUG:
            print('iteration_value:', self.iteration_value)
            print('iteration_value_list:', self.iteration_value_list)
            print('mean_value:', self.mean_value)
            
        return self.iteration_value
    
    def criterion_(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Criterion pass

        Args:
            p (Union[np.ndarray, torch.Tensor]): Predictions
            t (Union[np.ndarray, torch.Tensor]): Targets

        Returns:
            torch.Tensor: Loss
        """
        return self.criterion(p, torch.argmax(t, dim=1))
        # return self.criterion(p, t)
        
        
class AccuracyMetrics(Metrics):
    """ Accuracy Metrics based on the Metrics Class """
    def update(self, p: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor], convert_target2onehot=False) -> Union[torch.Tensor, np.ndarray]:
        """ Updates Loss

        Args:
            p (Union[np.ndarray, torch.Tensor]): Predictions
            t (Union[np.ndarray, torch.Tensor]): Targets
            convert_target2onehot (bool, optional): Target is in form of One-Hot Encoding. Defaults to False.

        Raises:
            TypeError: Predictions and Targets must be of type torch.tensor or np.ndarray

        Returns:
            Union[torch.Tensor, np.ndarray]: Accuracy
        """
        super().update(p, t, convert_target2onehot=convert_target2onehot)
        
        if convert_target2onehot:
            t = F.one_hot(t, p.shape[1])
        
        if self.DEBUG:
            print('Target:', t)
            
        if isinstance(p, torch.Tensor) and isinstance(t, torch.Tensor):
            matches = (torch.argmax(p, axis=1) == torch.argmax(t, axis=1)).cpu().numpy()
        elif isinstance(p, np.ndarray) and isinstance(t, np.ndarray):
            matches = (np.argmax(p, axis=1) == np.argmax(t, axis=1))
        else:
            raise TypeError(f'Predictions p:{type(p)} and Targets t:{type(t)} must both be of type torch.tensor or np.ndarray')
            
        self.iteration_value = matches.sum() / matches.shape[0]
        
        self.iteration_value_list.append(self.iteration_value)
        self.mean_value = np.array(self.iteration_value_list).mean(dtype=float)
        
        if self.DEBUG:
            print('matches:', matches)
            print('iteration_value:', self.iteration_value)
            print('iteration_value_list:', self.iteration_value_list)
            print('mean_value:', self.mean_value)
        
        return self.iteration_value
    
    
    
class TripletLossMetrics(LossMetrics):
    """ Triplet Loss Metrics based on the Metrics Class """
    def __init__(self, criterion, classification_criterion=nn.CrossEntropyLoss(), DEBUG: bool = False, *args, **kwargs) -> None:
        self.classification_criterion = classification_criterion

        super().__init__(criterion=criterion, DEBUG=DEBUG, *args, **kwargs)


    def criterion_(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Criterion pass with TripletLoss

        Args:
            p (Union[np.ndarray, torch.Tensor]): Predictions
            t (Union[np.ndarray, torch.Tensor]): Targets

        Returns:
            torch.Tensor: Loss
        """
        embeddings = p[0]        
        logits = p[1]
        
        triplet_loss = self.criterion(embeddings=embeddings, labels=t.argmax(1))
        loss = self.classification_criterion(logits, t.argmax(1))
        
        return triplet_loss + loss
        



# TPR=0
# TNR=0
# FPR=0
# FNR=0

# Recall = TPR/(TPR+FNR)
# Precision = TPR/(TPR+FPR)
# F1 = 2*Precision*Recall/(Precision+Recall)
# Accuracy = TPR+TNR/(TPR+TNR+FPR+FNR)

#ENDFILE
