'''Contains the loss functions used in the project'''
import torch

def cross_entropy_loss() -> torch.Tensor:
    '''Compute the cross-entropy loss'''
    return torch.nn.CrossEntropyLoss()
