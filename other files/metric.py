'''
This module contains the Mean Average Precision MAP metric for evaluating ranking algorithms
'''
import numpy as np

class MAP:
    ''' Mean Average Precision (MAP) metric for evaluating ranking algorithms'''

    @staticmethod
    def precision_at_k(y_true, y_pred, k=12):
        """
        Computes Precision at k for one sample.

        Parameters
        __________
        y_true: np.array
                Array of correct recommendations (Order doesn't matter)
        y_pred: np.array
                Array of predicted recommendations (Order does matter)
        k: int, optional
           Maximum number of predicted recommendations

        Returns
        _______
        score: double
               Precision at k
        """
        intersection = np.intersect1d(y_true, y_pred[:k])
        return len(intersection) / k

    @staticmethod
    def rel_at_k(y_true, y_pred, k=12):
        """
        Computes Relevance at k for one sample.

        Parameters
        __________
        y_true: np.array
                Array of correct recommendations (Order doesn't matter)
        y_pred: np.array
                Array of predicted recommendations (Order does matter)
        k: int, optional
           Maximum number of predicted recommendations

        Returns
        _______
        score: double
               Relevance at k
        """
        if y_pred[k-1] in y_true:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def average_precision_at_k(y_true, y_pred, k=12):
        """
        Computes Average Precision at k for one sample.

        Parameters
        __________
        y_true: np.array
                Array of correct recommendations (Order doesn't matter)
        y_pred: np.array
                Array of predicted recommendations (Order does matter)
        k: int, optional
           Maximum number of predicted recommendations

        Returns
        _______
        score: double
               Average Precision at k
        """
        ap = 0.0
        for i in range(1, k+1):
            ap += MAP.precision_at_k(y_true, y_pred, i) * MAP.rel_at_k(y_true, y_pred, i)
        return ap / min(k, len(y_true))

    def mean_average_precision(self, y_true, y_pred, k=10) -> float:
        """
        Computes MAP at k.

        Parameters
        __________
        k: int, optional
           Maximum number of predicted recommendations

        Returns
        _______
        score: double
               MAP at k
        """
        return np.mean([MAP.average_precision_at_k(gt, pred, k) for gt, pred in zip(y_true, y_pred)])