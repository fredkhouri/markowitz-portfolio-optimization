def calculate_averages(element_wise_results):
    """
    average[target_return] = mean over windows of OOS portfolio return.
    """
    return {
        target_return: float(np.mean(values))
        for target_return, values in element_wise_results.items()
    }
  def calculate_weighted_covariances(weights_matrices, oos_covariances, target_returns, verbose=False):
    """
    For each target_return and window i:
        var_i = w_i^T * Sigma_i * w_i

    We:
    - precompute a list of cov matrices in the correct order
    - avoid recomputing list(oos_covariances.keys()) each time
    """
    weighted_covariances = {target_return: [] for target_return in target_returns}

    # Fix the order of OOS windows once
    labels = list(oos_covariances.keys())
    cov_list = [np.asarray(oos_covariances[label]) for label in labels]

    for target_return, W in weights_matrices.items():
        # W shape: (n_windows, n_assets)
        for i, w in enumerate(W):
            cov = cov_list[i]               # (n_assets, n_assets)
            var = float(w @ cov @ w)        # scalar: portfolio variance for this window
            weighted_covariances[target_return].append(var)

        if verbose:
            print(f"Target {target_return}: first 5 OOS variances: "
                  f"{weighted_covariances[target_return][:5]}")

    return weighted_covariances


def store_weighted_covariances(weighted_covariance_results):
    """
    Just convert lists → numpy arrays per target_return.
    """
    return {
        target_return: np.array(results)
        for target_return, results in weighted_covariance_results.items()
    }
