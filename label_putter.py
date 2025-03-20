from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class LabelPutter(BaseEstimator, ClassifierMixin):
    """
    Обёртка для регрессора, превращающая его в классификатор
    """
    def __init__(self, regressor, classify_func: callable, *args, **kwargs):
        self.classify_func = classify_func
        self.regressor = regressor
        self.regressor.set_params(*args, **kwargs)
        self._is_fitted = False

    def fit(self, X, y):
        fit_result = self.regressor.fit(X, y)
        self._is_fitted = True
        if hasattr(self.regressor, "classes_"):
            self.classes_= self.regressor.classes_
        else:
            self.classes_ = np.unique(y)
        return fit_result
        
    
    def predict(self, X):
        return np.vectorize(self.classify_func)(self.regressor.predict(X))
    
    def get_params(self, *args, **kwargs):
         return {**self.regressor.get_params(*args, **kwargs), "regressor": self.regressor, "classify_func": self.classify_func}

    def set_params(self, *args, **kwargs):
        if "regressor" in kwargs:
            self.regressor = kwargs["regressor"]
            del kwargs["regressor"]
        if "classify_func" in kwargs:
            self.classify_func = kwargs["classify_func"]
            del kwargs["classify_func"]
        self.regressor.set_params(*args, **kwargs)
        return self
    
    def __sklearn_is_fitted__(self):
        return self._is_fitted


