"""
PBE exchange energy of a hydrogenic 1s orbital
via PySCF's bundled libxc (GGA_X_PBE functional).

Two equivalent approaches are demonstrated:
  1. scipy.quad + point-wise libxc call  – highest adaptive accuracy
  2. Dense radial grid + vectorial libxc  – faster for repeated evaluations

The density is fixed (not SCF): rho(r) = occ * |psi(r)|^2
with psi(r) = sqrt(zeta^3/pi) * exp(-zeta*r).

PySCF version : bundled libxc 7.0.0+
"""

import numpy as np
from scipy import integrate
from pyscf.dft import libxc

# ── Parameters ──────────────────────────────────────────────────────────────
zeta = 1.0   # Slater exponent
occ  = 1.0      # Occupation (adjustable)

# ── Helpers ──────────────────────────────────────────────────────────────────
def density_and_gradient(r: np.ndarray | float):
    """Return (rho, |grad rho|) for the 1s hydrogenic orbital at radius r."""
    psi      = np.sqrt(zeta**3 / np.pi) * np.exp(-zeta * r)
    dpsi_dr  = -zeta * psi
    rho      = occ * psi**2
    grad_mag = 2.0 * occ * np.abs(psi) * np.abs(dpsi_dr)
    return rho, grad_mag


def exc_pbe(rho: float, grad_mag: float) -> float:
    """
    PBE exchange energy density per particle via libxc (GGA_X_PBE).

    PySCF's eval_xc accepts spin-0 GGA density as shape-(4, N) array:
      row 0: rho
      row 1: drho/dx  (we treat the radial gradient as the x-component)
      row 2: drho/dy  (zero – purely radial orbital)
      row 3: drho/dz  (zero)
    Internally libxc computes sigma = |grad rho|^2 = row1^2 + row2^2 + row3^2.
    """
    rho_in       = np.zeros((4, 1))
    rho_in[0, 0] = rho
    rho_in[1, 0] = grad_mag          # |grad rho| along x  →  sigma = grad_mag^2
    exc, _vxc, _fxc, _kxc = libxc.eval_xc("GGA_X_PBE", rho_in, spin=0, deriv=1)
    return float(exc[0])


# ── Approach 1: scipy.quad with point-wise libxc ─────────────────────────────
def integrand_quad(r: float, occ: float, zeta: float) -> float:
    rho, grad_mag = density_and_gradient(r)
    if rho < 1e-300:
        return 0.0
    return rho * exc_pbe(rho, grad_mag) * 4.0 * np.pi * r**2


result_quad, error_quad = integrate.quad(
    integrand_quad, 0.0, 20.0,
    args=(occ, zeta),
    limit=200, epsabs=1e-12, epsrel=1e-12,
)

print("=" * 60)
print("Approach 1 – scipy.quad + point-wise libxc")
print("=" * 60)
print(f"  E_x(PBE)       = {result_quad:.15f} Ha")
print(f"  error estimate = {error_quad:.2e}")

# ── Approach 2: dense radial grid + vectorial libxc ──────────────────────────
r_grid           = np.linspace(1e-8, 25.0, 200_000)
rho_arr, grad_arr = density_and_gradient(r_grid)

# Build density input only where rho is non-negligible (avoids libxc warnings)
mask           = rho_arr > 1e-300
rho_in_vec     = np.zeros((4, int(mask.sum())))
rho_in_vec[0]  = rho_arr[mask]
rho_in_vec[1]  = grad_arr[mask]

exc_masked, _vxc, _fxc, _kxc = libxc.eval_xc(
    "GGA_X_PBE", rho_in_vec, spin=0, deriv=1
)

exc_full       = np.zeros(len(r_grid))
exc_full[mask] = exc_masked

integrand_grid = rho_arr * exc_full * 4.0 * np.pi * r_grid**2
result_grid    = np.trapezoid(integrand_grid, r_grid)

print()
print("=" * 60)
print("Approach 2 – dense radial grid + vectorial libxc")
print("=" * 60)
print(f"  E_x(PBE)       = {result_grid:.15f} Ha")

# ── Sanity check 1: norm of density ──────────────────────────────────────────
norm_integrand = rho_arr * 4.0 * np.pi * r_grid**2
norm_val       = np.trapezoid(norm_integrand, r_grid)

print()
print("=" * 60)
print("Sanity check 1 – normalisation of rho")
print("=" * 60)
print(f"  ∫ rho d³r  = {norm_val:.15f}  (should be {occ:.1f})")

# ── Sanity check 2: reduced gradient s at sample radii ───────────────────────
print()
print("=" * 60)
print("Sanity check 2 – reduced gradient s at sample r values")
print("=" * 60)
print(f"  {'r':>5}  {'rho':>12}  {'s':>12}  {'exc':>14}")
for r_val in [0.1, 0.5, 1.0, 2.0]:
    rho_pt, grad_pt = density_and_gradient(float(r_val))
    kf  = (3.0 * np.pi**2 * rho_pt) ** (1.0 / 3.0)
    s   = grad_pt / (2.0 * kf * rho_pt)
    eps = exc_pbe(rho_pt, grad_pt)
    print(f"  r={r_val:.1f}  rho={rho_pt:.6e}  s={s:.6f}  exc={eps:.8f}")
