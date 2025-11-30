def element_wise_multiplication(weights_matrices, oos_means, verbose=False):
    """
    For each target_return and each window i:
        result_i = sum_j w_{i,j} * mu_{i,j}
    where w = weights, mu = OOS means.

    This is now vectorized with einsum instead of Python loops.
    """
    oos_means_arr = oos_means.to_numpy()      # shape: (n_windows, n_assets)
    results = {}

    for target_return, W in weights_matrices.items():
        # W shape: (n_windows, n_assets)
        # per-window dot: sum_j W[i,j] * oos_means_arr[i,j]
        rw = np.einsum('ij,ij->i', W, oos_means_arr)  # vectorized over all windows
        results[target_return] = rw.tolist()

        if verbose:
            print(f"Target {target_return}: first 5 OOS portfolio returns: {rw[:5]}")

    return results
