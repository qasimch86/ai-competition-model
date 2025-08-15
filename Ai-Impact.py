import numpy as np
import matplotlib.pyplot as plt

def run_simulation(fast_training=False):
    # Horizon
    T = 20
    dt = 0.05
    n = int(T/dt)

    # Tech dynamics
    gamma_A = 0.6
    gamma_C = 0.4
    gamma_S = 0.1 if not fast_training else 0.6
    k_A = k_C = k_S = 1.0

    # NEW: depreciation so C,S don't saturate
    delta_C = 0.03
    delta_S = 0.02
    S_base = 0.0  # long-run baseline skill stock absent training

    # Job flows
    sigma = 0.25                 # AI displacement
    eta = 0.2                    # creation efficiency
    rho, kappa, tau = 0.5, 0.7, 0.8
    psi = 2.0                    # NEW: saturation exponent for creation

    # Matching
    m_H = 0.3 if not fast_training else 0.5
    m_N = 0.25 if not fast_training else 0.4

    # NEW: background separations and hiring cap
    s0 = 0.015                   # ~1.5% / year steady churn
    h_max = 0.08                 # max annual hiring/absorption fraction

    # State
    A = np.zeros(n); C = np.zeros(n); S = np.zeros(n)
    x_R = np.zeros(n); x_H = np.zeros(n); x_N = np.zeros(n); u = np.zeros(n)
    # Initial conditions
    S[0] = S_base+0.3  # initial skill level
    u[0] = 0.05  # initial unemployment
    x_N[0] = 0.1  # initial new-job share
    x_H[0] = 0.1  # initial hires share 
    x_R[0] = 0.8  # initial routine job share
    A[0] = 0.05  # initial tech level
    C[0] = 0.05  # initial capability level

    for t in range(n-1):
        # Tech evolution (with decay on C,S)
        dA = gamma_A * (1 - A[t]/k_A) * A[t]
        dC = gamma_C * A[t] * (1 - C[t]/k_C) - delta_C * C[t]
        dS = gamma_S * C[t] * (1 - S[t]/k_S) - delta_S * (S[t] - S_base)

        # Flows
        displacement = sigma * A[t] * x_R[t]
        # NEW: saturating creation -> slows late-stage absorption
        employed_share = x_H[t] + x_N[t]
        creation = eta * (A[t]**rho) * (C[t]**kappa) * (S[t]**tau) * ((1 - employed_share)**psi)

        # Matches (still simple & linear in u but capped later)
        match_H = m_H * u[t] * S[t]
        match_N = m_N * u[t] * C[t]

        # NEW: cap total hiring this period
        hiring_flow = min(h_max, match_H + match_N + creation)

        # Unemployment inflow includes background separations
        inflow_u = s0 * (1 - u[t]) + displacement
        outflow_u = hiring_flow
        du = inflow_u - outflow_u

        # Allocate hires priority: keep same weights as before
        total_intent = (match_H + match_N + creation) + 1e-12
        w_H = (match_H) / total_intent
        w_N = (match_N + creation) / total_intent  # treat creation as N-type
        hires_H = hiring_flow * w_H
        hires_N = hiring_flow * w_N

        # Employment stocks
        dx_R = -displacement - s0 * x_R[t]                  # routine loses to AI + churn
        dx_H = hires_H - s0 * x_H[t]                        # H gains via hires, loses to churn
        dx_N = hires_N - s0 * x_N[t]                        # N gains via hires, loses to churn

        # Integrate
        A[t+1] = np.clip(A[t] + dA*dt, 0, 1)
        C[t+1] = max(C[t] + dC*dt, 0)
        S[t+1] = max(S[t] + dS*dt, 0)

        x_R[t+1] = max(min(x_R[t] + dx_R*dt, 1.0), 0.0)
        x_H[t+1] = max(min(x_H[t] + dx_H*dt, 1.0), 0.0)
        x_N[t+1] = max(min(x_N[t] + dx_N*dt, 1.0), 0.0)

        # Keep totals in [0,1]
        emp_total = x_R[t+1] + x_H[t+1] + x_N[t+1]
        emp_total = min(emp_total, 1.0)
        u[t+1] = max(0.0, 1.0 - emp_total)  # implied by shares

        # Apply unemployment ODE with inflow/outflow too (blend to ensure floor)
        u[t+1] = max(0.0, min(1.0, u[t] + du*dt))

    return {
        "time": np.linspace(0, T, n),
        "A": A, "C": C, "S": S, "u": u,
        "x_R": x_R, "x_H": x_H, "x_N": x_N
    }

# --- Compare baseline vs fast training (same plots as before) ---
base = run_simulation(False)
fast = run_simulation(True)

fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axs[0].plot(base["time"], base["S"], label="Skills - Baseline")
axs[0].plot(fast["time"], fast["S"], label="Skills - Fast Training", linestyle="--")
axs[0].set_ylabel("Skill level"); axs[0].legend(); axs[0].set_title("Smoother tail with churn, capacity, and saturation")

axs[1].plot(base["time"], base["u"], label="Unemp. - Baseline")
axs[1].plot(fast["time"], fast["u"], label="Unemp. - Fast Training", linestyle="--")
axs[1].set_ylabel("Unemployment"); axs[1].legend()

axs[2].plot(base["time"], base["x_N"], label="New Jobs - Baseline")
axs[2].plot(fast["time"], fast["x_N"], label="New Jobs - Fast Training", linestyle="--")
axs[2].set_ylabel("New-job share"); axs[2].set_xlabel("Years"); axs[2].legend()

plt.tight_layout(); plt.show()
