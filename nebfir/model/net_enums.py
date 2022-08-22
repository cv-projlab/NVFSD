
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, SparseAdam
from torch.optim.lr_scheduler import (CosineAnnealingLR, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)

from ..env import *
from .i3d_network_arch import InceptionI3d
from .losses import TripletLoss
from .triplet_loss import BatchAllTtripletLoss

#############################################################################################################################################
################################################################ models ENUM ################################################################
#############################################################################################################################################

available_models=[InceptionI3d]

try:
    from mvit.models.mvit_model import MViT
    available_models.append(MVit)
except ImportError:
    warnings.warn(f'Package: mvit not installed ! Please check https://github.com/facebookresearch/mvit github page to install package')


try:
    from timesformer.models.vit import TimeSformer    
    available_models.append(TimeSformer)
except ImportError:
    warnings.warn(f'Package: timesformer not installed ! Please check https://github.com/facebookresearch/TimeSformer github page to install package')


models = Enum('models', dict(zip(list(map(lambda m: m.__name__, available_models)), available_models)))

#############################################################################################################################################
############################################################## criterions ENUM ##############################################################
#############################################################################################################################################
    
class criterions(Enum):
    ce = nn.CrossEntropyLoss
    triplet = nn.TripletMarginLoss
    # triplet_eurico = TripletLoss
    batch_all_triplet = BatchAllTtripletLoss
    
#############################################################################################################################################
############################################################## optimizers ENUM ##############################################################
#############################################################################################################################################

class optimizers(Enum):
    sgd = SGD
    adam = Adam
    adamw = AdamW
    sparse_adam = SparseAdam

#############################################################################################################################################
############################################################## schedulers ENUM ##############################################################
#############################################################################################################################################

class schedulers(Enum):
    plateau = ReduceLROnPlateau
    multi_step = MultiStepLR
    step = StepLR
    cosine_annealing = CosineAnnealingLR




# ENDFILE
