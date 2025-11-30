def rolling_mean_with_step_fast(df, window, step):
    """
    Optimized rolling mean using NumPy stride tricks.
    Fully equivalent to your original rolling_mean_with_step().
    """

    # DataFrame → numpy array
    X = df.to_numpy()
    n_rows, n_cols = X.shape

    # How many windows?
    n_windows = (n_rows - window) // step + 1

    # Strides (no copies)
    stride0, stride1 = X.strides
    windows = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, window, n_cols),
        strides=(step * stride0, stride0, stride1),
        writeable=False
    )

    # Compute each window mean
    means = windows.mean(axis=1)  # shape: (n_windows, n_cols)

    # Index of each window
    index = list(range(0, n_rows - window + 1, step))

    # Return perfect replacement
    return pd.DataFrame(means, index=index, columns=df.columns)
