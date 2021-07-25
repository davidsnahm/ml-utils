import numpy as np

def run(X, y, loss_fn="MSE", lr=0.1, num_iter=1000):
    w = np.random.randn(X.shape[1])
    grad_fn = get_gradient(loss_fn)
    for _ in range(num_iter):
        step = lr * grad_fn(X, y, w)
        w = w - step
        if abs(step) < 0.001:
            break
    return w

def get_gradient(loss_fn):
    if loss_fn == "MSE":
        def grad(X, y, w):
            return -2 / len(X) * np.dot(X, y - np.dot(X, w))
        return grad