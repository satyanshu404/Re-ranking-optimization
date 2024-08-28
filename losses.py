'''Contains the loss functions used in the project'''
import torch


def cross_entropy_loss() -> torch.Tensor:
    '''Compute the cross-entropy loss'''
    return torch.nn.CrossEntropyLoss()

class CrossEntropyLoss:
    '''Compute the cross-entropy loss'''
    def __init__(self):
        pass

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        '''Compute the softmax of a tensor'''
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)  # subtract max for numerical stability
        sum_x = torch.sum(exp_x, dim=1, keepdim=True)
        return exp_x / sum_x
    
    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        '''Compute the log softmax of a tensor'''
        return torch.log(self.softmax(x))

    def __call__(self, y_pred, y_true):
        """
        Compute the cross-entropy loss.

        Parameters:
        - y_pred: Tensor of shape (N, C) with the predicted logits (before softmax).
        - y_true: Tensor of shape (N,) with the ground truth labels (class indices).

        Returns:
        - loss: The cross-entropy loss as a scalar tensor.
        """

        batch_size = y_pred.shape[0]

        outputs = self.log_softmax(y_pred)
        outputs = outputs[range(batch_size), y_true]
        
        loss = -torch.sum(outputs) / batch_size
        return loss
