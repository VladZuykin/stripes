from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing import Callable
from numpy_typing import np


class StripeImplicit(BaseEstimator):
    def __init__(self, weights: np.float64_1d = None, step_function: float | Callable = 1, eps: float = .5,
                 corrections_max=np.inf, _is_fitted=False):
        """
        Инициализация
        :param weights: изначальные веса
        :param step_function: мультипликатор шага
        :param eps: максимум невязки
        :param corrections_max: максимальное количество исправлений
        :return: None
        """
        self.weights = weights
        self.step_function = step_function
        self.eps = eps
        self.corrections_max = corrections_max
        self.corrections_made = None
        self._param_names = ["weights", "step_function", "eps", "corrections_max", '_is_fitted']
        self._is_fitted = _is_fitted

    def fit(
            self,
            X: np.float64_2d,
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

            features_norm_square = features.dot(features)

            if callable(self.step_function):
                step_multiplier = self.step_function(corrections_made)
            elif isinstance(self.step_function, (float, int)):
                step_multiplier = self.step_function
            else:
                raise ValueError("Bad step function type.")

            future_weights = self.weights - (features.dot(self.weights) - answer) / (
                    0.5 / step_multiplier + features_norm_square) * features

            future_weights_residual = features.dot(future_weights) - answer

            gradient = 2 * features * future_weights_residual

            beta_k = future_weights_residual**2 + step_multiplier / 2 * gradient.dot(gradient)

            if beta_k > self.eps:
                self.weights = future_weights
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

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
