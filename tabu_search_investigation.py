import numpy as np
from tqdm import tqdm
import os
from Utils import *
from Tabu_Search import Tabu_Search

# This file used to investigate simultaneously the effect of MTM and STM size and step size on the performance of Tabu Search
tabu_2d_invest_dir = "TS_2D/"
if not os.path.exists(tabu_2d_invest_dir):
    os.makedirs(tabu_2d_invest_dir)

# Define parameters
intensify = np.linspace(5, 50, 8, dtype=int)
diversify = np.trunc(intensify*1.3)
ssr_trs = np.trunc(intensify*2.5)
step_sizes = np.linspace(80, 400, 8)

bound = 500
dims = 6
step_size = 100
ssr = 0.95

all_f_vals_list = []
for i in range(len(intensify)):
    for j in range(len(step_sizes)):
        print("Starting iteration: ", i, j)
        for k in range(30):
            f_vals = []
            algo = Tabu_Search(max_iter=15000, step_size=step_sizes[j], bound=bound, dims=dims, ssr_tr=ssr_trs[i], intensify_tr=intensify[i], diversify_tr=diversify[i], ssr_redu_factor=ssr, len_stm=20, len_mtm=30, grid_num=4, conv_step_size=5, dmin=50, dsim=5, a_lim=10)
            algo.main_search()
            f_vals.append(algo.best[0])
        all_f_vals_list.append(np.mean(f_vals))

fvalslist = np.array(all_f_vals_list).reshape(len(intensify),len(step_sizes))
plot_fn_results(intensify, step_sizes, fvalslist, 'intensify', 'step size', "TS_2D_search_FVALS.png", tabu_2d_invest_dir)