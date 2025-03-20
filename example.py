from implcit_stripe import StripeImplicitRegressor
from stripe import StripeRegressor
import numpy as np
from label_putter import LabelPutter


X = np.array(
        [
            [0, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1]
                ])
y = np.array([-1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1])

stripe = StripeRegressor(eps=0.01)
stripe.fit(X, y)
print("Weights:", stripe.weights)
print("Stripe regressor predictions:", stripe.predict(np.array([[0, 1], [1, 1]])))
stripe_classifier = LabelPutter(stripe, lambda x: 1 if x > 0 else -1)
print("Stripe classifier predictions:", stripe_classifier.predict(np.array([[0, 1], [1, 1]])))



def gen_samples(n=1000, rand=True):
        from random import randint
        x = []
        y = []
        for i in range(n):
            if rand:
                take = randint(0, 1)
            else:
                take = i % 2
            if take == 1:
                x.append([0, 1])
                y.append(-1)
            else:
                x.append([1, 1])
                y.append(1)
        return np.array(x), np.array(y)

implicit_stripe = StripeImplicitRegressor(eps=0.001, step_function=0.001)
implicit_stripe.fit(*gen_samples(20000))
print("Implicit stripe weights:", stripe.weights)
print("Implicit stripe regressor predictions:", stripe.predict(np.array([[0, 1], [1, 1]])))
print("Implicit stripe classifier predictions:", LabelPutter(implicit_stripe, lambda x: 1 if x > 0 else -1).predict(np.array([[0, 1], [1, 1]])))