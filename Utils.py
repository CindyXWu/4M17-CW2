import numpy as np
import matplotlib.pyplot as plt

DIMS = 2
BOUND = 500

def f(x):
    """Vectorised Schwefel function for one n-dimensional input vector.
    :param x: input vector
    """
    return -np.dot(np.sin(np.sqrt(np.abs(x))), x)

def penalty_f(x):
    """Vectorised Schwefel function with linear penalty.
    :param x: input vector for function to be evaluated on
    :param iter: iteration function is evaluated
    :param bound: -bound <= x_i <= bound
    """
    w = 50*np.ones(DIMS)
    # Pick out elements of array x > 500 in absolute value
    idx = np.argwhere(np.abs(x)<500)
    w[idx] = 0
    c_v = np.abs(np.array(x))-BOUND
    return f(x) + w.T@c_v

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
    x = np.linspace(-600, 600, 1000)
    y = np.linspace(-600, 600, 1000)
    X, Y = np.meshgrid(x, y)
    Z = list(map(lambda x: penalty_f(x), zip(X.flatten(), Y.flatten())))
    Z = np.array(Z).reshape((1000, 1000))
    plt.contour(X, Y, Z, 15, cmap='viridis')
    plt.colorbar()
    plt.savefig(title, dpi=300)
    plt.show()

def plot_fn_results(x, y, fevals, xlabel, ylabel, title, output_dir):
    """Plot function value varying over 2 parameters."""
    X, Y = np.meshgrid(x, y)
    #Z = list(map(lambda x: penalty_f(x), zip(X.flatten(), Y.flatten())))
    #Z = np.array(Z).reshape((1000, 1000))
    plt.contourf(X, Y, fevals, 20, cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.savefig(output_dir+title, dpi=300)
    plt.show()

def plot_results(x, y, xlabel, ylabel, title, output_dir):
    """Plot average function value or variance as a function of some parameter"""
    fig = plt.figure(figsize=(8,5))
    ax  = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(output_dir+title)

# if __name__ == "__main__":
#     x = [1, 2, 3]
#     y = [4, 5, 6]
#     fevals = np.random.uniform(0, 10, 9).reshape(3, 3)
#     plot_fn_results(x, y, fevals, "test.png")
