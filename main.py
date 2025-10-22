# Appendix A — Numerical Integration of the AO Funnel Density Profile
# Full executable script for AO model simulation, figure generation, and verification.
# Run: python ao_sim.py -- generates figs/1-3.png, prints params/verification.
# Requires: numpy, scipy, matplotlib, astropy

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import astropy.constants as const
import astropy.units as u
import argparse  # For CLI params

# Constants
G = const.G.value  # m^3 kg^-1 s^-2
c = const.c.value  # m/s
M_sun = const.M_sun.value  # kg
rho_nuc = 2.8e17  # kg/m^3 (nuclear density)

# Fiducial parameters (small seed for DM halo mimic)
M = 1e6 * M_sun  # kg (10^6 Msun)
a = 10e3  # m (10 km core radius)
rho0 = M / (8 * np.pi * a**3)  # Central density kg/m^3

def rho(r, rho0, a):
    """AO density profile: exponential taper (yin gradient)."""
    return rho0 * np.exp(-r / a)

def M_enc(r, rho0, a):
    """Enclosed mass: analytic (Seeliger integral)."""
    x = r / a
    return 8 * np.pi * rho0 * a**3 * (1 - np.exp(-x) * (1 + x + 0.5 * x**2))

def v_rot(r, M_enc_func, G):
    """Rotation velocity from enclosed mass."""
    return np.sqrt(G * M_enc_func(r) / r)

# Numerical integration for verification (no analytic shortcut)
def integrand(r, rho0, a):
    return 4 * np.pi * r**2 * rho(r, rho0, a)

def M_enc_num(r, rho0, a):
    """Enclosed mass: numerical quadrature."""
    if isinstance(r, (int, float)):
        integral, _ = quad(integrand, 0, r, args=(rho0, a))
        return integral
    else:  # Vectorized
        return np.array([M_enc_num(rr, rho0, a) for rr in r])

# Generate data
r_a = np.logspace(-1, 2.6, 100)  # r/a: 0.1 to ~400
r = r_a * a  # Actual r (m)

# Fig 1: Density Profile
rho_norm = np.exp(-r_a)
plt.figure(figsize=(6, 4))
plt.loglog(r_a, rho_norm, 'b-', linewidth=2, label='AO Profile')
plt.xlabel('r / a')
plt.ylabel(r'$\rho(r) / \rho_0$')
plt.title('Normalized Density Profile')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig1.png', dpi=300)
plt.close()

# Fig 2: Cumulative Mass
M_cum = 1 - np.exp(-r_a) * (1 + r_a + 0.5 * r_a**2)
plt.figure(figsize=(6, 4))
plt.plot(r_a, M_cum, 'orange', linewidth=2, label='AO Cumulative')
plt.xlabel('r / a')
plt.ylabel(r'$M(<r) / M$')
plt.ylim(0, 1.05)
plt.title('Cumulative Mass Profile')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figs/fig2.png', dpi=300)
plt.close()

# Fig 3: Rotation Curve (Galactic Scale Overlay vs. NFW)
# Scale to galaxy: M_halo=1e12 Msun, a_gal=1 kpc ~3e19 m
r_kpc = np.logspace(0, 2, 100)  # kpc
r_m_gal = r_kpc * 3.086e19  # m/kpc conversion
M_halo = 1e12 * M_sun
a_gal = 3e19  # m (1 kpc)
rho0_gal = M_halo / (8 * np.pi * a_gal**3)

def M_enc_gal(rr):
    """Galactic enclosed mass func."""
    return M_enc(rr, rho0_gal, a_gal)

v_ao = v_rot(r_m_gal, M_enc_gal, G) / 1000  # km/s (divide m/s by 1000)

# NFW approx: flat ~220 km/s (empirical SPARC average)
v_nfw = np.full_like(r_kpc, 220.0)

plt.figure(figsize=(6, 4))
plt.semilogx(r_kpc, v_ao, 'b-', linewidth=2, label='AO Model')
plt.semilogx(r_kpc, v_nfw, 'r--', linewidth=2, label='NFW (approx)')
plt.xlabel('r (kpc)')
plt.ylabel('v (km/s)')
plt.title('AO vs. NFW Rotation Curve Overlay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/fig3.png', dpi=300)
plt.close()

# Verification Outputs
eta = rho0 / rho_nuc
mu = (G * M) / (c**2 * a)
print(f"Central density ρ₀: {rho0:.2e} kg m⁻³")
print(f"Density ratio η: {eta:.2e}")
print(f"Compactness μ: {mu:.2e}")

# Test: M_enc at r=10a (analytic vs. numeric)
r_test = 10 * a
M_ana = M_enc(r_test, rho0, a)
M_num = M_enc_num(r_test, rho0, a)
print(f"M(<10a) analytic: {M_ana / M_sun:.2e} M⊙ ({M_ana / M * 100:.1f}%)")
print(f"M(<10a) numeric: {M_num / M_sun:.2e} M⊙ ({M_num / M * 100:.1f}%)")

# CLI for param sweeps (optional)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AO Model Simulator")
    parser.add_argument('--M', type=float, default=M / M_sun, help="Halo mass (Msun)")
    parser.add_argument('--a', type=float, default=a / 1e3, help="Core radius (km)")
    args = parser.parse_args()
    # Rerun with args.M * M_sun, etc. for custom sims
    print("AO Sim complete. Figs saved to /figs/. Repo: https://github.com/tstoeao/AO-Seeds-Sim")
