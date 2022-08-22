from ..env import *
from ..metrics.metrics_new import LossMetrics, AccuracyMetrics, TripletLossMetrics

class metrics_enum(Enum):
    accuracy_metric = AccuracyMetrics
    
    loss_metric = LossMetrics
    triplet_loss_metric = TripletLossMetrics
    
