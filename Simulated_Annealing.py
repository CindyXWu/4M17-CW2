import numpy as np
import matplotlib.pyplot as plt
import timeit
import heapq
from operator import itemgetter
from tqdm import tqdm
from Utils import *
import os

output_dir = "SA/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class Simulated_Annealing():
    def __init__(self, init_prob, bound, chain_length, step_size, eta, cool_rate, max_iter, final_temp, min_acceptance_prob, dmin, dsim, a_lim):
        """"Some parameters initialised. Constant throughout algorithm are denoted by capitals."""
        # Iteration business
        self.iteration = 0
        self.k = 0  # Number of temperature steps taken
        self.max_iter = max_iter    # Maximum number of iterations allowed (break in case stopping criteria not met)
        self.f_evals = 0

        # Fundamental parameters for annealing
        self.DIMS = len(bound)
        self.T = np.inf # Set initial T to infinity
        self.x = np.random.uniform(-bound, bound, self.DIMS)   # Initial x random in permissible space
        self.f_val = self.f(self.x)
        self.f_new = np.inf
        self.sigma = np.inf # Standard deviation of objective function values accepted at latest temperature T_k

        # Other parameters for variations of algo
        self.STEP_SIZE = step_size  # Step size for fixed step size update in update_x
        self.PROB = init_prob
        self.BOUND = bound
        self.CHAIN_LENGTH = chain_length    # Maximum Markov chain length at kth temperature
        self.ETA = eta  # Minimum trials accepted at kth temperature - typically 0.6*chain_length
        self.MIN_ACCEPT_PROB = min_acceptance_prob    # Minimum acceptance probability for stopping criteria
        self.cool_rate = cool_rate  # Cooling rate for annealing schedule
        self.final_temp = final_temp    # Final temperature for stopping criteria
        # For multiple stopping criteria possible pick one which comes first

        # Vanderbilt and Louie update parameters
        self.Q = self.STEP_SIZE*np.eye(self.DIMS) # Initial covariance matrix for VL update
        self.ALPHA = 0.1   # Damping constant for VL update
        self.OMEGA = 2.1  # Weighting factor for VL update
        self.N_ITER = 50    # Number of steps to evaluate X in VL update

        # Archiving/ solution saving
        self.DMIN = dmin
        self.DSIM = dsim
        self.ALIM = a_lim    # Archive size
        self.best_archive = []   # Archive for best solutions; list of tuples: (f(x), x)
        self.within_dmin = []   # Values in archive within dmin of current solution (dynamically updated)
        self.within_dsim = []   # Values in archive within dsim of current solution (dynamically updated)
        self.accepted_archive_k = []    # Accepted x values at kth temperature (dynamically updated)
        self.accepted_archive_k_f = []  # Accepted f(x) values at kth temperature (dynamically updated)
        self.historic_archive = [] # Archive of all accepted solutions in history of algo (used only for 2D plots)
        self.historic_archive_f = [] # Archive of all accepted f(x) values in history of algo (used only for 2D plots)
    
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
    
    def update_x(self):
        """Update x to new point using fixed step size and random uniform vector.
        Check for out of bounds proposals so can be used without penalty function.

        :param step: vector of step sizes (elementwise)
        """
        u = np.random.uniform(-1, 1, self.DIMS)
        x_new = self.x + self.STEP_SIZE*u
        while any(np.abs(x_i) > self.BOUND[i] for i, x_i in enumerate(x_new)):
            u = np.random.uniform(-1, 1, self.DIMS)
            x_new = self.x + self.STEP_SIZE*u
        return x_new
    
    def update_x_VL(self):
        """Update x using method presented in section 2.1.2 of report.

        :param S_init: initial covariance matrix
        :param alpha: damping constant
        :param omega: weighting factor
        :param n_iter: number of steps to evaluate X
        :return q_new: new covariance based on local area
        """
        proposals = np.empty((self.N_ITER, self.DIMS))
        for i in range(self.N_ITER):
            u = self.get_u_VL()
            x_new = self.x + self.Q @ u
            while any(np.abs(x_i) > self.BOUND[i] for i, x_i in enumerate(x_new)):
                u = self.get_u_VL()
                x_new = self.x + self.Q @ u
            proposals[i,:] = x_new
        # Take covariance of each of M 6D vectors in proposals
        cov = np.cov(proposals, rowvar=False)
        s_old = self.Q@self.Q.T
        s_new = self.ALPHA*self.OMEGA*cov + (1-self.ALPHA)*s_old
        self.Q = np.linalg.cholesky(s_new)
        x_new = self.x + self.Q @ self.get_u_VL()
        while any(np.abs(x_i) > self.BOUND[i] for i, x_i in enumerate(x_new)):
            x_new = self.x + self.Q @ self.get_u_VL()
        return x_new

    def get_u_VL(self):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), self.DIMS)

    def find_T0_Johnson(self, n_iter, update_func):
        """"
        Find starting temperature depending on initial probability of acceptance.
        Sets class variable T.

        :param x0: initial n-dim vector
        :param n_iter: num evals of function to calculate average increase
        """
        delta_es = []
        while len(delta_es) < n_iter:
            x_new = update_func()
            self.f_new = self.f(x_new)
            if self.f_new > self.f_val:
                delta_es.append(self.f_new - self.f_val)
        delta_e = np.mean(delta_es)
        self.T = -delta_e/np.log(self.PROB)
    
    def find_T0_White(self, n_iter, update_func):
        """"
        Find starting temperature depending on initial probability of acceptance.
        Sets class variable T."""
        return
    
    def accepted(self):
        """Calculate acceptance probability and return True/False specific solution."""
        delta_f = self.f_new - self.f_val
        if delta_f < 0:
            return True
        else:
            return np.random.uniform() < np.exp(-delta_f/self.T)

    def cooling(self):
        """Exponential decrement cooling schedule."""
        self.T = self.cool_rate*self.T
    
    def adaptive_cooling_Huang(self):
        """Adaptive cooling schedule."""
        self.sigma = np.sqrt(np.var(self.accepted_archive_k_f))
        self.cool_rate = max(0.5, np.exp(-0.7*self.T/self.sigma))
        self.T = self.cool_rate*self.T

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
            heapq._heapreplace_max(self.best_archive, [self.f_val, self.x])
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
                self.best_archive.remove(max(self.within_dsim, key=itemgetter(0)))
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
    
    def restart(self):
        """Restart the search from the best solution found so far."""
        self.x = min(self.best_archive,key=itemgetter(0))[1]
    
    def plot_2D(self, points, title):
        """Plot 2D function."""
        fig, ax = plt.subplots(figsize=(12,10),dpi=300)
        x = np.linspace(-500, 500, 1000)
        y = np.linspace(-500, 500, 1000)
        X, Y = np.meshgrid(x, y)
        Z = list(map(lambda x: self.f(x), zip(X.flatten(), Y.flatten())))
        Z = np.array(Z).reshape((1000, 1000))
        plt.contour(X, Y, Z, 15, cmap='viridis')
        t = np.arange(len(points))
        plt.scatter([x[0] for x in points], [x[1] for x in points], c=t, s=10)
        plt.colorbar()
        plt.savefig(output_dir+title, dpi=300)

    def main_search(self, update_func):
        """Main SA search loop method. Note kwargs allows update func to be passed relevant parameters."""
        # Set initial temperature
        self.find_T0_Johnson(100, update_func)

        # Outer loop
        # First met condition causes algorithm break
        # Note only 15000 function evaluations allowed
        while self.T > self.final_temp and self.iteration < self.max_iter and self.f_evals < 14000:
            iters_at_temp = 0

            # First met condition causes temperature cooling
            # Break conditions: max trials evaluated at temp, max accepted solutions
            while iters_at_temp < self.CHAIN_LENGTH and len(self.accepted_archive_k) < self.ETA:
                x_new = update_func()
                self.f_new = self.f(x_new)
                # Markov chain length depends on number of evaluations, not acceptances
                iters_at_temp += 1
                if self.accepted():
                    self.x = x_new
                    self.f_val = self.f_new
                    self.accepted_archive_k.append(self.x)
                    self.accepted_archive_k_f.append(self.f_val)
                    # Perform archiving
                    self.archive()

            # Restart if no solutions accepted at temperature, but keep temp same   
            if not self.accepted_archive_k:
                self.restart()
                continue

            self.adaptive_cooling_Huang()  
            self.historic_archive.extend(self.accepted_archive_k)
            self.historic_archive_f.extend(self.accepted_archive_k_f)
            self.accepted_archive_k = []    # Clear accepted solutions archive for temperature
        
        # Get best solution sorted by function value in archive
        self.best = sorted(self.best_archive, key=itemgetter(0))[0]   # Best solution

        # Plotting for 2D
        if self.DIMS == 2:
            self.plot_2D(self.historic_archive, "2D_VL_update_adaptcool_Huang.png")
            xs = [data[1] for data in self.best_archive]
            self.plot_2D(xs, "2D_VL_update_adaptcool_Huang_archive.png")

# bound = np.array([500, 500])
# cl = 100
# step_size = 1000
# final_temp = 1
# algo1 = Simulated_Annealing(init_prob=0.8, bound=bound, chain_length=cl, step_size=step_size, eta=0.6*cl, cool_rate=0.95, max_iter=15000, final_temp=final_temp, min_acceptance_prob=0.001, dmin=50, dsim=5, a_lim=10)

if __name__ == "__main__":

    # # ========================== chain length testing ==========================
    # all_runtimes = []
    # all_f_vals = []
    # all_sol_variances = []
    # all_best_solutions = []
    # all_f_vals_variances = []
    # all_f_eval_nums = []

    # low_cl_archive = []
    # low_cl_evol = []
    # high_cl_archive = []
    # high_cl_evol = []
    
    # # Edit chain lengths to test
    # chain_lengths = np.logspace(1, 3, 8, dtype=int)
    # cl_dir = "SA_CL/"
    # if not os.path.exists(cl_dir):
    #     os.makedirs(cl_dir)

    # # Bound dimensions defines dimensionality of problem (no separate specification to avoid clashes)
    # bound = np.array([500, 500, 500])
    # dims = len(bound)
    # step_size = 1000
    # final_temp = 1

    # # Run 50 times for each chain length
    # for cl in chain_lengths:
    #     runtimes = []
    #     solutions = []
    #     f_vals = []
    #     dim_vars = []
    #     f_eval_nums = []
    #     for i in tqdm(range(30)):
    #         algo1 = Simulated_Annealing(init_prob=0.8, bound=bound, chain_length=cl, step_size=step_size, eta=0.6*cl, cool_rate=0.95, max_iter=15000, final_temp=final_temp, min_acceptance_prob=0.001, dmin=50, dsim=5, a_lim=10)
    #         start_time = timeit.default_timer()
    #         algo1.main_search(algo1.update_x_VL)
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
    #     if cl == 10:
    #         low_cl_archive = [data[0] for data in algo1.best_archive]
    #         low_cl_evol = algo1.historic_archive_f
    #     elif cl == 1000:
    #         high_cl_archive = [data[0] for data in algo1.best_archive]
    #         high_cl_evol = algo1.historic_archive_f

    # np.savetxt(cl_dir+"function_evolution_cl10.csv", np.array(low_cl_evol), delimiter=",")
    # np.savetxt(cl_dir+"function_evolution_cl100.csv", np.array(high_cl_evol), delimiter=",")
    # np.savetxt(cl_dir+"best_archive_cl10.csv", np.array(low_cl_archive), delimiter=",")
    # np.savetxt(cl_dir+"best_archive_cl100.csv", np.array(high_cl_archive), delimiter=",")
    # np.savetxt(cl_dir+"runtimes.csv", np.array(all_runtimes), delimiter=",")
    # np.savetxt(cl_dir+"bestsols.csv", np.array(all_best_solutions), delimiter=",")
    # np.savetxt(cl_dir+"solvariances.csv", np.array(all_sol_variances), delimiter=",")
    # np.savetxt(cl_dir+"chain_lengths.csv", np.array(chain_lengths), delimiter=",")
    # np.savetxt(cl_dir+"f_vals.csv", np.array(all_f_vals), delimiter=",")
    # np.savetxt(cl_dir+"f_vals_variances.csv", np.array(all_f_vals_variances), delimiter=",")
    # np.savetxt(cl_dir+"f_eval_nums.csv", np.array(all_f_eval_nums), delimiter=",")
    # plot_results(np.arange(0, len(low_cl_evol)), low_cl_evol, "Accepted move number", "Function Value", "function_evolution_cl10", cl_dir)
    # plot_results(np.arange(0, len(high_cl_evol)), high_cl_evol, "Accepted move number", "Function Value", "function_evolution_cl100", cl_dir)
    # plot_results(chain_lengths, all_runtimes, "Chain Length", "Runtime (s)", "Runtime", cl_dir)
    # plot_results(chain_lengths, all_best_solutions, "Chain Length", "Average Solution Control Variable Value", "Average Solution Control Variable Value", cl_dir)
    # plot_results(chain_lengths, all_sol_variances, "Chain Length", "Average Control Variable Variance", "Average Control Variable Variance", cl_dir)
    # plot_results(chain_lengths, all_f_vals, "Chain Length", "Average Function Value", "Average Function Value", cl_dir)
    # plot_results(chain_lengths, all_f_vals_variances, "Chain Length", "Variance in Function Value", "Variance in Function Value", cl_dir)


    #========================= step size ==========================
    all_runtimes = []
    all_f_vals = []
    all_sol_variances = []
    all_best_solutions = []
    all_f_vals_variances = []
    all_f_eval_nums = []

    low_ss_archive = []
    low_ss_evol = []
    high_ss_archive = []
    high_ss_evol = []
    
    step_sizes = np.logspace(1, 2, 3)
    step_size_dir = "SA_SS/"
    if not os.path.exists(step_size_dir):
        os.makedirs(step_size_dir)

    # Bound dimensions defines dimensionality of problem (no separate specification to avoid ssashes)
    bound = np.array([500, 500, 500, 500, 500, 500])
    dims = len(bound)
    chain_length = 100
    final_temp = 1

    # Run 50 times for each chain length
    for ss in step_sizes:
        runtimes = []
        solutions = []
        f_vals = []
        dim_vars = []
        f_eval_nums = []
        for i in tqdm(range(2)):
            algo1 = Simulated_Annealing(init_prob=0.8, bound=bound, chain_length=chain_length, step_size=ss, eta=0.6*ss, cool_rate=0.95, max_iter=15000, final_temp=final_temp, min_acceptance_prob=0.001, dmin=50, dsim=5, a_lim=10)
            start_time = timeit.default_timer()
            algo1.main_search(algo1.update_x_VL)
            runtimes.append(timeit.default_timer() - start_time)
            # Add mean value of all dimensions in solution found (i.e. take mean over dims AND runs)
            solutions.append(np.mean(algo1.best[1]))
            f_vals.append(algo1.best[0])
            f_eval_nums.append(algo1.f_evals)
        all_runtimes.append(np.mean(np.array(runtimes)))
        all_best_solutions.append(np.mean(np.array(solutions)))
        # Mean variance of all dimensions in solution found
        all_sol_variances.append(np.mean(np.array(solutions)))
        all_f_vals_variances.append(np.var(np.array(f_vals)))
        all_f_vals.append(np.mean(np.array(f_vals)))
        all_f_eval_nums.append(np.mean(np.array(f_eval_nums)))
        if ss == 10:
            low_ss_archive = [data[0] for data in algo1.best_archive]
            low_ss_evol = algo1.historic_archive_f
        elif ss == 100:
            high_ss_archive = [data[0] for data in algo1.best_archive]
            high_ss_evol = algo1.historic_archive_f

    np.savetxt(step_size_dir+"function_evolution_ss10.csv", np.array(low_ss_evol), delimiter=",")
    np.savetxt(step_size_dir+"function_evolution_ss100.csv", np.array(high_ss_evol), delimiter=",")
    np.savetxt(step_size_dir+"best_archive_ss10.csv", np.array(low_ss_archive), delimiter=",")
    np.savetxt(step_size_dir+"best_archive_ss100.csv", np.array(high_ss_archive), delimiter=",")
    np.savetxt(step_size_dir+"runtimes.csv", np.array(all_runtimes), delimiter=",")
    np.savetxt(step_size_dir+"bestsols.csv", np.array(all_best_solutions), delimiter=",")
    np.savetxt(step_size_dir+"solvariances.csv", np.array(all_sol_variances), delimiter=",")
    np.savetxt(step_size_dir+"step_sizes.csv", np.array(step_sizes), delimiter=",")
    np.savetxt(step_size_dir+"f_vals.csv", np.array(all_f_vals), delimiter=",")
    np.savetxt(step_size_dir+"f_vals_variances.csv", np.array(all_f_vals_variances), delimiter=",")
    np.savetxt(step_size_dir+"f_eval_nums.csv", np.array(all_f_eval_nums), delimiter=",")
    plot_results(np.arange(0, len(low_ss_evol)), low_ss_evol, "Accepted move number", "Function Value", "function_evolution_ss10", step_size_dir)
    plot_results(np.arange(0, len(high_ss_evol)), high_ss_evol, "Accepted move number", "Function Value", "function_evolution_ss100", step_size_dir)
    plot_results(step_sizes, all_runtimes, "Chain Length", "Runtime (s)", "Runtime",  step_size_dir)
    plot_results(step_sizes, all_best_solutions, "Chain Length", "Average Solution Control Variable Value", "Average Solution Control Variable Value", step_size_dir)
    plot_results(step_sizes, all_sol_variances, "Chain Length", "Average Control Variable Variance", "Average Control Variable Variance", step_size_dir)
    plot_results(step_sizes, all_f_vals, "Chain Length", "Average Function Value", "Average Function Value", step_size_dir)
    plot_results(step_sizes, all_f_vals_variances, "Chain Length", "Variance in Function Value", "Variance in Function Value", step_size_dir)

# ===================================== Cooling rate against final temperature =====================================


