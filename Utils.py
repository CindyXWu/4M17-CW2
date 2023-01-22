import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Vectorised Schwefel function for one n-dimensional input vector.
    :param x: input vector
    """
    return -np.dot(np.sin(np.sqrt(np.abs(x))), x)

def penalty_f(x, bound, temp):
    """Vectorised Schwefel function with linear penalty.
    :param x: input vector for function to be evaluated on
    :param iter: iteration function is evaluated
    :param bound: -bound <= x_i <= bound
    """
    w = 100*np.ones(x.shape)
    # pick out elements of vector x that are less than 500
    w[x<500] = 0
    c_v = x-bound
    return f(x) + w.T@c_v/temp

def penalty_f_evolving(x, iter, mu, bound):
    """Vectorised Schwefel function with linear penalty.
    :param x: input vector for function to be evaluated on
    :param factor: factor>1 - optimisation more biased towards penalty as search goes on.
    :param iter: iteration function is evaluated at
    :param bound: -bound <= x_i <= bound
    """
    w = 100*np.ones(x.shape)*mu*iter
    # pick out elements of vector x that are less than 500
    w[x<500] = 0
    c_v = x-bound
    return f(x) + w.T@c_v

def plot_2D(title):
    """Plot 2D function."""
    x = np.linspace(-500, 500, 1000)
    y = np.linspace(-500, 500, 1000)
    X, Y = np.meshgrid(x, y)
    Z = list(map(lambda x: f(x), zip(X.flatten(), Y.flatten())))
    Z = np.array(Z).reshape((1000, 1000))
    plt.contour(X, Y, Z, 15, cmap='viridis')
    plt.colorbar()
    plt.savefig(title, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_2D("Schwefel_2D.png")