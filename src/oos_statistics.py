def oos_rolling_statistics(df, in_sample_window, oos_window, step):
    oos_means = {col: [] for col in df.columns}
    oos_covariances = {}
    indices = []
    for start in range(0, len(df) - in_sample_window - oos_window + 1, step):
        in_sample_end = start + in_sample_window
        oos_end = in_sample_end + oos_window
        if oos_end <= len(df):
            oos_window_data = df.iloc[in_sample_end:oos_end]
            oos_window_means = oos_window_data.mean()
            indices.append(f"window {len(indices) + 1}")
            for col in df.columns:
                oos_means[col].append(oos_window_means[col])
            oos_cov_matrix = oos_window_data.cov()
            oos_covariances[indices[-1]] = oos_cov_matrix
    oos_means_df = pd.DataFrame(oos_means, index=indices)
    return oos_means_df, oos_covariances
