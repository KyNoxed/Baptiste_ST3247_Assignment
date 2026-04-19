import numpy as np
from numba import njit

@njit
def simulate_numba(beta, gamma, rho, N, p_edge, n_infected0, T, seed=0):

    np.random.seed(seed)

    # Adjacency matrix
    adj = np.zeros((N, N), dtype=np.uint8)

    # Build Erdos-Renyi graph
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                adj[i, j] = 1
                adj[j, i] = 1

    # States: 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)

    # Initial infected
    perm = np.random.permutation(N)
    for k in range(n_infected0):
        state[perm[k]] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)

    infected_fraction[0] = np.sum(state == 1) / N

    for t in range(1, T + 1):

        # --------------------
        # PHASE 1: Infection
        # --------------------
        new_state = state.copy()

        for i in range(N):
            if state[i] == 1:
                for j in range(N):
                    if adj[i, j] == 1 and state[j] == 0:
                        if np.random.random() < beta:
                            new_state[j] = 1

        state = new_state

        # --------------------
        # PHASE 2: Recovery
        # --------------------
        for i in range(N):
            if state[i] == 1:
                if np.random.random() < gamma:
                    state[i] = 2

        # --------------------
        # PHASE 3: Rewiring
        # --------------------
        rewire_count = 0

        for i in range(N):
            if state[i] == 0:  # susceptible
                for j in range(N):
                    if adj[i, j] == 1 and state[j] == 1:

                        if np.random.random() < rho:

                            # remove edge
                            adj[i, j] = 0
                            adj[j, i] = 0

                            # find new partner
                            # brute-force sampling
                            attempts = 0
                            while attempts < N:
                                k = np.random.randint(0, N)
                                if k != i and adj[i, k] == 0:
                                    adj[i, k] = 1
                                    adj[k, i] = 1
                                    rewire_count += 1
                                    break
                                attempts += 1

        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    # --------------------
    # Degree histogram
    # --------------------
    degree_histogram = np.zeros(31, dtype=np.int64)

    for i in range(N):
        deg = 0
        for j in range(N):
            deg += adj[i, j]

        if deg >= 30:
            degree_histogram[30] += 1
        else:
            degree_histogram[deg] += 1

    return infected_fraction, rewire_counts, degree_histogram
