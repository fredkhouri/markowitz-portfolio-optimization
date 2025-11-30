def construct_q_matrix(cov_matrix, mean_returns):
    n = len(mean_returns)

    # Validate
    assert cov_matrix.shape == (n, n)
    
    # Create full matrix
    Q = np.zeros((n+2, n+2))

    # Top-left block (covariance)
    Q[:n, :n] = cov_matrix

    # Mean & ones (constraints)
    Q[:n, n]     = -mean_returns
    Q[:n, n + 1] = -1
    Q[n,   :n]   = -mean_returns
    Q[n+1, :n]   = -1

    return Q
