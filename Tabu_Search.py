import numpy as np
import matplotlib.pyplot as plt
import timeit
from operator import itemgetter
import os
from tqdm import tqdm
from Utils import *

output_dir = "TS/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class Tabu_Search():
    def __init__(self, max_iter, step_size, bound, dims, ssr_tr, intensify_tr, diversify_tr, ssr_redu_factor, len_stm, len_mtm, grid_num, conv_step_size, dmin, dsim, a_lim):
        """"Some parameters initialised. Constant throughout algorithm are denoted by capitals.
        :param grid_num: number of grid squares in each dimension
        :param conv_step_size: minimum step size defines convergence
        """
        # Iteration business
        self.iteration = 0
        self.MAX_ITER = max_iter    # Maximum number of iterations allowed (break in case stopping criteria not met)
        self.f_evals = 0

        # Fundamental parameters for tabu search
        self.DIMS = dims
        self.BOUND = bound  # Bound on search space
        self.STEP_SIZE = step_size  # (isotropic) step size for local search
        self.SSR_TRIGGER = ssr_tr  # Unsuccessful iteration number before step size reduction triggered
        self.I_TRIGGER = intensify_tr  # "" intensification
        self.D_TRIGGER = diversify_tr  # "" diversification
        self.STM_LEN = len_stm  # Max length of STM
        self.MTM_LEN = len_mtm  # Max length of MTM
        self.CONV_SS = conv_step_size  # Minimum step size defines convergence
        self.HALF_GRID = bound/grid_num  # Half the size of each grid square
        self.GRID_CENTRES = np.linspace(-bound+self.HALF_GRID, bound-self.HALF_GRID, grid_num)  # Centre of grid squares in 1D for discretisation of search space
        self.ssr_factor = ssr_redu_factor  # Factor by which step size is reduced when SSR is triggered
        self.counter = 0    # Counter for number of iterations of unsuccessful evaluation
        self.x = np.random.uniform(-bound, bound, self.DIMS)   # Initial x random in permissible space
        self.f_val = self.penalty_f(self.x)
        self.best_f_eval = np.inf # Best function value so far, used for triggering SSD
        self.best_x = np.inf   # Best solution so far
        self.to_change = [] # Index of element of x to change in local search. Used for prioritisation of control variables.

        # Solution saving
        self.STM = [self.x.tolist()]
        self.MTM = [[self.f_val, self.x]]  # List of tuples (f(x), x)
        self.DMIN = dmin
        self.DSIM = dsim
        self.ALIM = a_lim    # Archive size
        # Archive also acts as MTM
        self.best_archive = [[self.f_val, self.x]]   # Archive for best solutions; list of tuples: (f(x), x)
        self.within_dmin = []   # Values in archive within dmin of current solution (dynamically updated)
        self.within_dsim = []   # Values in archive within dsim of current solution (dynamically updated)
        self.historic_archive = [self.x] # Archive of all accepted solutions in history of algo (used only for 2D plots)
        self.historic_archive_f = [self.f_val] # Archive of all accepted f(x) values in history of algo (used only for 2D plots)
        d_grid = np.array([grid_num]*self.DIMS)
        self.grid_sols = np.zeros(d_grid)   # Save number of solutions in each grid square. Initialise as empty array with dimensions grid_num^dims
    
    def f(self, x):
        """Schwefel function for one n-dimensional input vector."""
        self.f_evals += 1
        return -np.dot(np.sin(np.sqrt(np.absolute(x))), x)
    
    def penalty_f(self, x):
        """Vectorised Schwefel function with linear penalty.
        :param x: input vector for function to be evaluated on
        :param iter: iteration function is evaluated
        :param bound: -bound <= x_i <= bound
        """
        self.f_evals += 1
        w = 50*np.ones(self.DIMS)
        # Pick out elements of array x < 500 in absolute value - do not apply penalty to these
        idx = np.argwhere(np.abs(x)<500)
        w[idx] = 0
        c_v = np.abs(np.array(x))-self.BOUND
        return self.f(x) + w.T@c_v
    
    def update_grid(self):
        """Update grid for storing number of solutions in each part of search space - 4**d sections"""
        # Full n-dim index list for storing solutions in grid, updated with loop below
        idxs = np.empty(self.DIMS, dtype=int)
        for i in range(self.DIMS):
            # Find grid sector with closest centre to current solution element
            idx = np.argmin(np.abs(self.GRID_CENTRES - self.x[i]))
            idxs[i] = idx
        # Update grid
        self.grid_sols[idxs] += 1
        
    def intensification(self):
        """Search intensification: move to average of MTM solutions"""
        MTM = np.array(self.MTM)
        self.x = np.mean(MTM[:,1], axis=0)

    def diversification(self):
        """Search diversification: move to randomly selected part of search space. Involves discretising search space into grid."""
        if not np.any(self.grid_sols):
            # If no solutions found yet, pick random point
            self.x = np.random.uniform(-self.BOUND, self.BOUND, self.DIMS)
            return
        # Pick out first grid square with least number of solutions
        idxs = np.argwhere(self.grid_sols == np.min(self.grid_sols))
        # Randomly pick one of these grid cubes
        idx = idxs[np.random.randint(0, len(idxs))]
        # Randomly pick point in that grid cube
        for i in range(self.DIMS):
            self.x[i] = np.random.uniform(self.GRID_CENTRES[idx[i]]-self.HALF_GRID, self.GRID_CENTRES[idx[i]]+self.HALF_GRID)
    
    def step_size_reduction(self):
        """If no best solution found for ssr_tr number of steps, reduce step size"""
        self.STEP_SIZE *= self.ssr_factor
        # Reset counter
        self.counter = 0

    def update_memories(self):
        """Update STM, MTM."""
        if len(self.STM) <= self.STM_LEN:
            self.STM.append(self.x.tolist())
        else:
            self.STM.pop(0)
            self.STM.append(self.x.tolist())
        # Update MTM (list of tuples [f_val, x])
        if len(self.MTM) < self.MTM_LEN:
            self.MTM.append([self.f_val, self.x])
        elif self.f_val < any([data[0] for data in self.MTM]):
            self.MTM.append([self.f_val, self.x])
            self.MTM.remove(max(self.MTM, key=itemgetter(0)))
    
    def archive(self):
        """Archive best previously found solutions."""
        # Updates list of similar solutions to current solution (within dsim)
        dissimilar = self.__dissimilar()

        # Archive not full and solution dissimilar to all present, add
        if len(self.best_archive) < self.ALIM and dissimilar:
            self.best_archive.append([self.f_val, self.x])
            return

        # Archive full + solution dissimilar to all present + better than largest f valued solution in archive, replace worst with current solution
        if len(self.best_archive) == self.ALIM and dissimilar and self.f_val < max(self.best_archive, key=itemgetter(0))[0]:
            self.best_archive.remove(max(self.best_archive, key=itemgetter(0)))
            self.best_archive.append([self.f_val, self.x])
            return

        # Archive full and x not dissimilar to solutions:
        if len(self.best_archive) == self.ALIM and not dissimilar:

            # Archive if best solution found so far and replace worst archived within dmin
            if self.f_val < min(self.best_archive,key=itemgetter(0))[0]:
                self.best_archive.append([self.f_val, self.x])
                self.best_archive.remove(max(self.within_dmin, key=itemgetter(0)))
                return

            # If not best solution found so far, but within dsim of some solution(s)
            if self.within_dsim:
                self.best_archive.append([self.f_val, self.x])
                self.best_archive.remove(max(self.within_dsim, key=lambda x: x[0]))
                return
    
    def __dissimilar(self):
        """Check if new solution is dissimilar to previously found solutions.
        Also update two lists of solutions within dsim and dmin of x. Assumes dmin > dsim.
        """
        dissimilar = True
        self.within_dsim = []  # List of solutions within dsim of x
        self.within_dmin = [] # List of solutions within dmin of x
        for data in self.best_archive:
            distance = np.linalg.norm(self.x - data[1])
            if distance < self.DMIN:
                dissimilar = False
                self.within_dmin.append(data)
                if distance < self.DSIM:
                    self.within_dsim.append(data)
        return dissimilar

    def plot_2D(self, points, title):
        """Plot 2D function."""
        fig, ax = plt.subplots(figsize=(12,10),dpi=300)
        x = np.linspace(-500, 500, 1000)
        y = np.linspace(-500, 500, 1000)
        X, Y = np.meshgrid(x, y)
        Z = list(map(lambda x: self.penalty_f(x), zip(X.flatten(), Y.flatten())))
        Z = np.array(Z).reshape((1000, 1000))
        plt.contour(X, Y, Z, 15, cmap='viridis')
        t = np.arange(len(points))
        plt.scatter([x[0] for x in points], [x[1] for x in points], c=t, s=10)
        plt.colorbar()
        plt.savefig(output_dir+title, dpi=300)
    
    def local_search(self):
        """Local search loop, including pattern move."""
        # Holds delta f values for each dimension's proposed move (increment and decrement)
        self.delta_fs = np.empty(2*self.DIMS)
        # Selection of which dimensions to change
        if len(self.to_change) == 0:
            dims_to_search = range(self.DIMS)
        else:
            dims_to_search = self.to_change

        for i in dims_to_search:
            new_x = self.x
            new_x[i] += self.STEP_SIZE
            # If new solution in STM or out of bounds, set delta f as infinity
            if new_x.tolist() in self.STM or any(np.abs(x_i) > self.BOUND for x_i in new_x):
                self.delta_fs[i] = np.inf
            else:
                self.delta_fs[i] = self.penalty_f(new_x) - self.f_val
            
            # Now for decrement
            new_x = self.x
            new_x[i] -= self.STEP_SIZE
            if  new_x.tolist() in self.STM or any(np.abs(x_i) > self.BOUND for x_i in new_x):
                self.delta_fs[i+self.DIMS] = np.inf
            else:
                self.delta_fs[i+self.DIMS] = self.penalty_f(new_x) - self.f_val

        if not all(np.isinf(self.delta_fs)):
            # Otherwise make move that reduces function most
            best_idx = np.argmin(self.delta_fs)
            move = np.zeros(self.DIMS)
            if best_idx < self.DIMS:
                idx = best_idx
                move[idx] += self.STEP_SIZE
            else:
                idx = best_idx - self.DIMS
                move[idx] -= self.STEP_SIZE

            # Pattern moves
            x_new = self.x + move
            x1 = x_new
            x2 = x1 + move
            f1 = self.penalty_f(x1)
            f2 = self.penalty_f(x2)
            while f2 < f1:
                x1 = x2
                f1 = f2
                x2 = x1 + move
                f2 = self.penalty_f(x2)
            self.x = x1
            self.f_val = f1
            return True
        else:
            return False
    
    def var_prioritise_local_search(self, frac):
        """For larger dimensions, searching all control variables gets messy.
        Prioritise searching control variables with greatest influence on objective function. 
        :param frac: Fraction of top varying control variables to search.
        """
        sensitivity = np.empty(self.DIMS)
        self.to_change = []
        f = self.f(self.x)
        for i in range(self.DIMS):
            new_x = self.x
            new_x[i] += self.STEP_SIZE
            plusf = self.f(new_x) - f
            # Now for decrement
            new_x = self.x
            new_x[i] -= self.STEP_SIZE
            minusf = self.f(new_x) - f
            sensitivity[i] = np.abs((plusf - minusf)/(2*self.STEP_SIZE))
        # Return indices that would sort the sensitivity array ascending
        idxs = np.argsort(sensitivity)
        # Pick highest sensitivity variables to search
        self.to_change = idxs[:int(self.DIMS*frac)]
        print(self.to_change)
        self.local_search()

    def main_search(self):
        """Main search loop"""
        while self.STEP_SIZE > self.CONV_SS and self.f_evals < 15000:

            new_sol = self.local_search()

            if new_sol:
                self.update_grid()
                self.update_memories()
                self.archive()
                self.historic_archive.append(self.x)
                self.historic_archive_f.append(self.f_val)
                # If new best solution found, reset counter and start local search
                if self.f_evals < self.best_f_eval:
                    self.best_f_eval = self.f_evals
                    self.best_x = self.x
                    self.counter = 0
                    continue

            # Else increment counter
            self.counter += 1
            # Check for search intensification, diversification, step size reduction
            if self.counter == self.I_TRIGGER:
                self.intensification()
            if self.counter == self.D_TRIGGER:
                self.diversification()
            if self.counter == self.SSR_TRIGGER:
                self.step_size_reduction()
                self.counter = 0
                continue
        # Get best solution sorted by function value in archive
        self.best = sorted(self.best_archive, key=itemgetter(0))[0]   # Best solution
        # Plotting for 2D
        if self.DIMS == 2:
            self.plot_2D(self.historic_archive, "2D_TS.png")
            xs = [data[1] for data in self.best_archive]
            self.plot_2D(xs, "2D_TS_archive.png")
        

if __name__ == "__main__":
    runtimes = []         
    algo1 = Tabu_Search(max_iter=15000, step_size=300, bound=500, dims=2, ssr_tr=25, intensify_tr=10, diversify_tr=15, ssr_redu_factor=0.9, len_stm=7, len_mtm=12, grid_num=4, conv_step_size=5, dmin=50, dsim=5, a_lim=10)
    algo1.main_search()
    print(algo1.historic_archive)
    print(algo1.MTM)

# if __name__ == "__main__":
    # all_runtimes = []
    # all_f_vals = []
    # all_sol_variances = []
    # all_best_solutions = []
    # all_f_vals_variances = []
    # all_f_eval_nums = []

    # low_ss_archive = []
    # low_ss_evol = []
    # high_ss_archive = []
    # high_ss_evol = []

    # # Bound dimensions defines dimensionality of problem (no separate specification to avoid clashes)
    # bound = 500
    # dims = 6
    # step_sizes = np.logspace(1, 2.7, 10)
    # step_size_dir = "TS_SS/"
    # if not os.path.exists(step_size_dir):
    #     os.makedirs(step_size_dir)

    # for ss in step_sizes:
    #     runtimes = []
    #     solutions = []
    #     f_vals = []
    #     dim_vars = []
    #     f_eval_nums = []
    #     for i in tqdm(range(2)):
    #         algo1 = Tabu_Search(max_iter=15000, step_size=ss, bound=bound, dims=dims, ssr_tr=100, intensify_tr=30, diversify_tr=50, ssr_redu_factor=0.95, len_stm=20, len_mtm=30, grid_num=4, conv_step_size=5, dmin=50, dsim=5, a_lim=10)
    #         start_time = timeit.default_timer()
    #         algo1.main_search()
    #         runtimes.append(timeit.default_timer() - start_time)
    #         # Add mean value of all dimensions in solution found (i.e. take mean over dims AND runs)
    #         solutions.append(np.mean(algo1.best[1]))
    #         f_vals.append(algo1.best[0])
    #         f_eval_nums.append(algo1.f_evals)
    #     all_runtimes.append(np.mean(np.array(runtimes)))
    #     all_best_solutions.append(np.mean(np.array(solutions)))
    #     # Variance of all dimensions in solution found (var will compute variance of flattened array)
    #     all_sol_variances.append(np.var(np.array(solutions)))
    #     all_f_vals_variances.append(np.var(np.array(f_vals)))
    #     all_f_vals.append(np.mean(np.array(f_vals)))
    #     all_f_eval_nums.append(np.mean(np.array(f_eval_nums)))
    #     filename = "function_evolution_step_size_"+str(ss)+".png"
    #     plot_results(np.arange(0, len(algo1.historic_archive_f)), algo1.historic_archive_f, "Accepted move number", "Function Value", filename, step_size_dir)
    
    # np.savetxt(step_size_dir+"runtimes.csv", np.array(all_runtimes), delimiter=",")
    # np.savetxt(step_size_dir+"bestsols.csv", np.array(all_best_solutions), delimiter=",")
    # np.savetxt(step_size_dir+"solvariances.csv", np.array(all_sol_variances), delimiter=",")
    # np.savetxt(step_size_dir+"step_sizes.csv", np.array(step_sizes), delimiter=",")
    # np.savetxt(step_size_dir+"f_vals.csv", np.array(all_f_vals), delimiter=",")
    # np.savetxt(step_size_dir+"f_vals_variances.csv", np.array(all_f_vals_variances), delimiter=",")
    # np.savetxt(step_size_dir+"f_eval_nums.csv", np.array(all_f_eval_nums), delimiter=",")
    # plot_results(step_sizes, all_runtimes, "Step Size", "Runtime (s)", "Runtime", step_size_dir)
    # plot_results(step_sizes, all_best_solutions, "Step Size", "Average Solution Control Variable Value", "Average Solution Control Variable Value", step_size_dir)
    # plot_results(step_sizes, all_sol_variances, "Step Size", "Average Control Variable Variance", "Average Control Variable Variance", step_size_dir)
    # plot_results(step_sizes, all_f_vals, "Step Size", "Average Function Value", "Average Function Value", step_size_dir)
    # plot_results(step_sizes, all_f_vals_variances, "Step Size", "Variance in Function Value", "Variance in Function Value", step_size_dir)

    # #========================================== step size reduction factor ==========================================
    # all_runtimes = []
    # all_f_vals = []
    # all_sol_variances = []
    # all_best_solutions = []
    # all_f_vals_variances = []
    # all_f_eval_nums = []

    # low_ss_archive = []
    # low_ss_evol = []
    # high_ss_archive = []
    # high_ss_evol = []

    # # Bound dimensions defines dimensionality of problem (no separate specification to avoid clashes)
    # bound = 500
    # dims = 6
    # step_size = 300
    # ssrs = np.linspace(0.6, 0.99, 10)
    # ssr_dir = "TS_SSRED/"
    # if not os.path.exists(ssr_dir):
    #     os.makedirs(ssr_dir)

    # for ssr in ssrs:
    #     runtimes = []
    #     solutions = []
    #     f_vals = []
    #     dim_vars = []
    #     f_eval_nums = []
    #     for i in tqdm(range(50)):
    #         algo1 = Tabu_Search(max_iter=15000, step_size=step_size, bound=bound, dims=dims, ssr_tr=100, intensify_tr=30, diversify_tr=50, ssr_redu_factor=ssr, len_stm=20, len_mtm=30, grid_num=4, conv_step_size=5, dmin=50, dsim=5, a_lim=10)
    #         start_time = timeit.default_timer()
    #         algo1.main_search()
    #         runtimes.append(timeit.default_timer() - start_time)
    #         # Add mean value of all dimensions in solution found (i.e. take mean over dims AND runs)
    #         solutions.append(np.mean(algo1.best[1]))
    #         f_vals.append(algo1.best[0])
    #         f_eval_nums.append(algo1.f_evals)
    #     all_runtimes.append(np.mean(np.array(runtimes)))
    #     all_best_solutions.append(np.mean(np.array(solutions)))
    #     # Mean variance of all dimensions in solution found
    #     all_sol_variances.append(np.mean(np.array(solutions)))
    #     all_f_vals_variances.append(np.var(np.array(f_vals)))
    #     all_f_vals.append(np.mean(np.array(f_vals)))
    #     all_f_eval_nums.append(np.mean(np.array(f_eval_nums)))
    #     plot_results(np.arange(0, len(algo1.historic_archive_f)), algo1.historic_archive_f, "Accepted move number", "Function Value", "function_evolution_ss_reduction_factor_"+str(ssr)+".png", ssr_dir)

    # np.savetxt(ssr_dir+"runtimes.csv", np.array(all_runtimes), delimiter=",")
    # np.savetxt(ssr_dir+"bestsols.csv", np.array(all_best_solutions), delimiter=",")
    # np.savetxt(ssr_dir+"solvariances.csv", np.array(all_sol_variances), delimiter=",")
    # np.savetxt(ssr_dir+"ssrs.csv", np.array(ssrs), delimiter=",")
    # np.savetxt(ssr_dir+"f_vals.csv", np.array(all_f_vals), delimiter=",")
    # np.savetxt(ssr_dir+"f_vals_variances.csv", np.array(all_f_vals_variances), delimiter=",")
    # np.savetxt(ssr_dir+"f_eval_nums.csv", np.array(all_f_eval_nums), delimiter=",")
    # plot_results(ssrs, all_runtimes, "Step Size Reduction Factor", "Runtime (s)", "Runtime", ssr_dir)
    # plot_results(ssrs, all_best_solutions, "Step Size Reduction Factor", "Average Solution Control Variable Value", "Average Solution Control Variable Value", ssr_dir)
    # plot_results(ssrs, all_sol_variances, "Step Size Reduction Factor", "Average Control Variable Variance", "Average Control Variable Variance", ssr_dir)
    # plot_results(ssrs, all_f_vals, "Step Size Reduction Factor", "Average Function Value", "Average Function Value", ssr_dir)
    # plot_results(ssrs, all_f_vals_variances, "Step Size Reduction Factor", "Variance in Function Value", "Variance in Function Value", ssr_dir)
