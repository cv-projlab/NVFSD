
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F



class Metrics:
    def __init__(self, loss_criterion) -> None:
        self.loss_criterion = loss_criterion

        self.reset()

    def reset(self):
        self.iteration_accuracy: float = 0.0
        self.accuracy_list: list = []
        self.mean_accuracy: float = 0.0

        self.iteration_loss: float = 0
        self.loss_list: list = []
        self.mean_loss: float = 0.0

    def compute(self, outputs, labels):
        assert (
            outputs.shape == labels.shape
        ), f"labels must be one hot vectors and have same shape as logits. labels shape:{labels.shape}; logits shape:{outputs.shape}"

        self.compute_loss(outputs=outputs, labels=labels)
        self.compute_accuracy(outputs=outputs, labels=labels)

    def compute_loss(self, outputs, labels):
        """ Calculates the cross entropy loss

            Args:
                outputs (torch.Tensor): logits with same shape as labels
                labels (torch.Tensor): one hot vector with labels 
            """
        # self.iteration_loss = self.loss_criterion(outputs, labels.squeeze()) # users -> ex: labels = [1,0,2]
        self.iteration_loss = self.loss_criterion(
            outputs, torch.max(labels, 1).indices
        )  # labels -> ex: labels = [[0,1,0],[1,0,0],[0,0,1]]; torch.max(labels, 1).indices = [1,0,2]
        self.loss_list.append(self.iteration_loss.item())
        self.mean_loss = np.array(self.loss_list).mean(dtype=float)

    def compute_accuracy(self, outputs, labels):
        """ Calculates the accuracy

            Args:
                outputs (torch.Tensor): logits with same shape as labels
                labels (torch.Tensor): one hot vector with labels 
            """
        matches = (torch.argmax(outputs.cpu(), 1) == torch.argmax(labels.cpu(), 1)).float()
        # matches = (torch.argmax(outputs.cpu(), 1) == torch.argmax(labels.cpu())).float()

        self.iteration_accuracy = matches.sum() / matches.shape[0]
        self.accuracy_list.append(self.iteration_accuracy.item())
        self.mean_accuracy = np.array(self.accuracy_list).mean()

    def get_accuracy(self):
        return self.iteration_accuracy

    def get_mean_accuracy(self):
        return self.mean_accuracy

    def get_loss(self):
        return self.iteration_loss

    def get_mean_loss(self):
        return self.mean_loss

