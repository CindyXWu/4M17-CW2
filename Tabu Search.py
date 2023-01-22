import numpy as np
import matplotlib.pyplot as plt
import timeit
import heapq
from operator import itemgetter
import os

output_dir = "TS/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class Tabu_Search():
    def __init__(self, max_iter):
        """"Some parameters initialised. Constant throughout algorithm are denoted by capitals."""
        # Iteration business
        self.iteration = 0
        self.max_iter = max_iter    # Maximum number of iterations allowed (break in case stopping criteria not met)
        self.f_evals = 0
    
    def f(self, x):
        """Schwefel function for one n-dimensional input vector."""
        self.f_evals += 1
        return -np.dot(np.sin(np.sqrt(np.absolute(x))), x)