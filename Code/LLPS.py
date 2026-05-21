import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load simulation parameters ---
with open('input.txt', 'r') as file:
    exec(file.read())

steps = int(t_final / dt)
save_interval = 10000
save_phi_interval = 500000  

# --- Initialize grid and field ---
np.random.seed(0)
phi0 = 0.5
noise = 0.01 * (2 * np.random.rand(N, N) - 1)
phi = np.clip(phi0 + noise, eps, 1 - eps)

# --- Initialize data storage ---
variance_list = []
time_list = []

# --- Helper functions ---
def laplacian(f):
    return (np.roll(f, +1, axis=0) + np.roll(f, -1, axis=0) +
            np.roll(f, +1, axis=1) + np.roll(f, -1, axis=1) -
            4 * f) / dx**2

def chi(c):
    return chi0 + A1 * c / (K1 + c) - A2 * c**2 / (K2**2 + c**2)

def dchi_dc(c):
    d1 = A1 * K1 / (K1 + c)**2
    d2 = A2 * (2 * c * K2**2) / (K2**2 + c**2)**2
    return d1 - d2

# --- Logging setup ---
log_file = open("phi_stats_log.txt", "w")
log_file.write("Step\tMin_phi\tMax_phi\tMean_phi\n")

# --- Time evolution loop ---
for step in range(steps):
    c = c0 * (1 - phi)
    chi_c = chi(c)
    dchi_dphi = -c0 * dchi_dc(c)

    phi_safe = np.clip(phi, eps, 1 - eps)

    mu = (1 / N1) * np.log(phi_safe) - np.log(1 - phi_safe)
    mu += chi_c * (1 - 2 * phi_safe) + dchi_dphi * phi_safe * (1 - phi_safe)
    mu -= kappa * laplacian(phi)

    phi += dt * M * laplacian(mu)
    phi = np.clip(phi, eps, 1 - eps)
    phi += (phi0 - phi.mean())  
    

    if np.isnan(phi).any():
        print(f"NaN encountered at step {step}, exiting.")
        break

    # --- Save image and statistics ---
    if step % save_interval == 0:
        plt.figure(figsize=(5, 5), dpi=300)
        im = plt.imshow(phi, origin='lower', cmap='plasma', vmin=0, vmax=1)
        plt.title(f'$t = {step*dt:.2f}$', fontsize=15)
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.tick_params(axis='both', labelsize=14)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.02)
        cbar.set_label('φ', fontsize=15)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout(pad=0.2)
        plt.savefig(f'au_{step//save_interval:04d}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Log statistics
        min_phi, max_phi, mean_phi = phi.min(), phi.max(), phi.mean()
        print(f"Step {step}, min φ = {min_phi:.4f}, max φ = {max_phi:.4f}, mean φ = {mean_phi:.4f}")
        log_file.write(f"{step}\t{min_phi:.6f}\t{max_phi:.6f}\t{mean_phi:.6f}\n")

        # Record variance
        variance_list.append(phi.var())
        time_list.append(step * dt)

    # --- Save φ field periodically ---
    if step % save_phi_interval == 0 and step != 0:
        np.save(f"phi_step_{step}.npy", phi)
        print(f"Saved φ field at step {step} to phi_step_{step}.npy")

log_file.close()

# --- Save variance vs time data ---
with open('variance_vs_time.txt', 'w') as f:
    f.write('Time\tVariance_phi\n')
    for t, v in zip(time_list, variance_list):
        f.write(f'{t:.6f}\t{v:.6f}\n')

# --- Plot variance evolution ---
plt.figure(figsize=(6, 5), dpi=300)
plt.plot(time_list, variance_list, linewidth=2)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Variance of φ', fontsize=16)
plt.title('Variance of φ vs Time', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('variance_vs_time.png', dpi=300)
plt.close()
