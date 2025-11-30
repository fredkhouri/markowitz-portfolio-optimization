def rolling_covariance_opt(df, window, step):
    arr = df.to_numpy()                       # Convert once → contiguous array
    n_rows, n_cols = arr.shape

    covariances = {}
    indices = []

    for start in range(0, n_rows - window + 1, step):
        window_arr = arr[start:start+window]  # contiguous slice → no copies

        # Compute covariance manually via NumPy (vectorized)
        X = window_arr - window_arr.mean(axis=0)
        cov = (X.T @ X) / (len(window_arr) - 1)

        covariances[start] = cov
        indices.append(start)

    return covariances
