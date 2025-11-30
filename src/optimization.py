from scipy.optimize import minimize
from joblib import Parallel, delayed
from numba import njit
import numpy as np

epsilon = 1e-6
target_returns = np.arange(0.005, 0.105, 0.005)

# ---------- JIT-compiled kernels (heavy math) ----------

@njit(fastmath=True)
def risk_function_jit(w, sigma):
    # w^T Sigma w
    tmp = sigma @ w
    return (w * tmp).sum()

@njit(fastmath=True)
def minimum_return_constraint_jit(w, target_return, mean_returns):
    # μ^T w - R_min
    return (w * mean_returns).sum() - target_return

@njit(fastmath=True)
def sum_to_one_constraint_jit(w):
    # 1^T w - 1
    return w.sum() - 1.0

# ---------- Thin Python wrappers for SciPy ----------

def risk_function(w, sigma):
    # SciPy calls this; inside is Numba-accelerated
    return risk_function_jit(w, sigma)

def minimum_return_constraint(w, target_return, mean_returns):
    return minimum_return_constraint_jit(w, target_return, mean_returns)

def sum_to_one_constraint(w):
    return sum_to_one_constraint_jit(w)


# Initial guess and bounds (unchanged)
initial_weights = [1/83] * 83
bounds = [(-2, 2)] * 83 


def rolling_window_optimization(rolling_means, rolling_covariances, target_returns,
                                n_jobs=-1):

    n_assets = rolling_means.shape[1]

    # Build all optimization tasks: one per (window_start, target_return)
    tasks = []
    for start in rolling_means.index:
        mean_returns = rolling_means.loc[start].values.astype(np.float64)

        cov_obj = rolling_covariances[start]
        sigma = cov_obj.values.astype(np.float64) if hasattr(cov_obj, "values") else np.asarray(cov_obj, dtype=np.float64)

        for target_return in target_returns:
            tasks.append((start, mean_returns, sigma, float(target_return)))

    total_iterations = len(tasks)

    # Worker: solve ONE optimization problem
    def solve_one(start, mean_returns, sigma, target_return):
        constraints = (
            {
                'type': 'eq',
                'fun': minimum_return_constraint,
                'args': (target_return, mean_returns)
            },
            {
                'type': 'eq',
                'fun': sum_to_one_constraint
            }
        )

        result = minimize(
            risk_function,
            initial_weights,
            args=(sigma,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return start, target_return, result.x

    # Run all tasks in parallel (joblib)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(solve_one)(start, mean_returns, sigma, target_return)
        for (start, mean_returns, sigma, target_return) in tasks
    )

    # Rebuild the 'solutions' dict in the same structure as before
    solutions = {}
    for start, target_return, weights in results:
        if start not in solutions:
            solutions[start] = {}
        solutions[start][target_return] = weights

    print(f"Completed {total_iterations} optimizations (parallel, JIT-accelerated).")

    return solutions

