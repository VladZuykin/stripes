from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from numpy_typing import np


class StripeRegressor(BaseEstimator):
    def __init__(self, weights: np.float64_1d = None, step_multiplier:float=1, eps: float=.5, corrections_max=np.inf, _is_fitted=False):
        """
        Инициализация
        :param weights: изначальные веса
        :param step_multiplier: мультипликатор шага
        :param eps: максимум невязки
        :param corrections_max: максимальное количество исправлений
        :return: None
        """
        self.weights = weights
        self.step_multiplier = step_multiplier
        self.eps = eps
        self.corrections_max = corrections_max
        self.corrections_made = None
        self._param_names = ["weights", "step_multiplier", "eps", "corrections_max", '_is_fitted']
        self._is_fitted = _is_fitted

    def fit(
            self,
            X: np.float64_1d,
            y: np.float64_1d
    ):
        """
        Настройка весов
        :param X: данные
        :param y: ответы
        """

        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        corrections_made = 0
        for row in np.concatenate((X, np.resize(y, (y.shape[0], 1))), axis=1):
            features, answer = row[:-1], row[-1]
            residual = features.dot(self.weights) - answer
            if (np.abs(residual) >= self.eps) and corrections_made <  self.corrections_max:
                self.weights = self.weights - self.step_multiplier * residual / np.linalg.norm(features, ord=2) ** 2 * features
                corrections_made += 1
                if corrections_made >= self.corrections_max:
                    break
        self.corrections_made = corrections_made
        self._is_fitted = True
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        return X.dot(self.weights)
    
    def __sklearn_is_fitted__(self):
        return self._is_fitted
    
    def get_params(self, deep = True):
         return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
