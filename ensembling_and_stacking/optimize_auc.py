import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics


class OptimizeAUC:
    """
    Class for optimizing AUC.
    This class is all you need to find best weights for any model
    and for any metric and for any types of predictions.
    With very small changes, this class can be used for optimization of 
    weights in ensemble models of _any_ type of predictions
    """
    def __init__(self):
        self.coef_ = 0
    
    def _auc(self, coef: list, X: np.array, y: np.array) -> float:
        # multiply coefficients with every column of the array with predictions
        x_coef = X * coef

        # create predictions by taking row wise sum
        predictions = np.sum(x_coef, axis=1)

        # return negative auc
        score = -1 * metrics.roc_auc_score(y, predictions)
        return score
    
    def fit(self, X: np.array, y: np.array) -> None:
        loss_partial = partial(self._auc, X=X, y=y)

        # dirichlet distribution.
        # you can use any distribution you want to initialize the coefficients
        # we want the coefficients to sum to 1
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        # use scipy fmin to minimize the loss function
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)
    
    def predict(self, X: np.array) -> np.array:
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

