import numpy as np
import matplotlib.pyplot as plt
import timeit
import heapq
from operator import itemgetter
import os
from Utils import *
from Tabu_Search import Tabu_Search

# This file used to investigate simultaneously the effect of 
tabu_2d_invest_dir = "TS_SS/"
if not os.path.exists(step_size_dir):
    os.makedirs(step_size_dir)