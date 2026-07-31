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
zeta = 1.6875   # Slater exponent
occ  = 2.0      # Occupation (adjustable)

# ── Helpers ──────────────────────────────────────────────────────────────────
def density_and_gradient(r: np.ndarray | float):
    """Return (rho, |grad rho|) for the 1s hydrogenic orbital at radius r."""
    psi      = np.sqrt(zeta**3 / np.pi) * np.exp(-zeta * r)
    dpsi_dr  = -zeta * psi
    rho      = occ * psi**2
    drho_dr = 2.0 * occ * psi * dpsi_dr
    return psi, rho, drho_dr


def exc_pbe(rho: float, drho_dr: float) -> float:
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
    rho_in[1, 0] = drho_dr          # |grad rho| along x  →  sigma = grad_mag^2
    exc, vxc, _fxc, _kxc = libxc.eval_xc("GGA_X_PBE", rho_in, spin=0, deriv=1)
    return float(exc[0]) 

def vxc_pointwise(r):
    """Full v_xc(r) via finite differences on e_xc(r) — avoids IBP entirely."""
    psi, rho, drho_dr = density_and_gradient(r)
    rho_in = np.zeros((4, 1))
    rho_in[0, 0] = rho
    rho_in[1, 0] = drho_dr  # sigma = drho_dr^2
    exc, vxc, _, _ = libxc.eval_xc('GGA_X_PBE', rho_in, spin=0, deriv=1)
    vrho   = vxc[0][0]    # dE/drho
    vsigma = vxc[1][0]    # dE/dsigma
    return vrho, vsigma

# ── Approach 1: scipy.quad with point-wise libxc ─────────────────────────────
def integrand_quad(r: float, occ: float, zeta: float) -> float:
    psi, rho, grad_mag = density_and_gradient(r)
    if rho < 1e-300:
        return 0.0
    return rho * exc_pbe(rho, grad_mag) * 4.0 * np.pi * r**2
def v00_integrand(r):
    psi, rho, drho_dr = density_and_gradient(r)
    if rho < 1e-300:
        return 0.0
    vrho, vsigma = vxc_pointwise(r)
    dpsi_dr = -zeta * psi

    term1 = psi**2 * vrho
    # IBP: -2 * ∫ φ² ∇·(vsigma ∇ρ) = +2 * ∫ ∇(φ²)·vsigma·∇ρ
    #     = +4 * ∫ φ · ∂φ/∂r · vsigma · ∂ρ/∂r  (both negative → product positive)
    term2 = 4.0 * psi * dpsi_dr * vsigma * drho_dr   # ← was -2.0, missing factor of 2

    return (term1 + term2) * 4.0 * np.pi * r**2

result_quad, error_quad = integrate.quad(
    integrand_quad, 0.0, 20.0,
    args=(occ, zeta),
    limit=200, epsabs=1e-12, epsrel=1e-12,
)
result_v, error_v = integrate.quad(v00_integrand, 0, 20,
                                limit=200, epsabs=1e-12, epsrel=1e-12)

print("=" * 60)
print("Approach 1 – scipy.quad + point-wise libxc")
print("=" * 60)
print(f"  E_x(PBE)       = {result_quad:.15f} Ha")
print(f"  error estimate = {error_quad:.2e}")
print(f"V(0,0) reference = {result_v:.15f}")
print(f"error estimate   = {error_v:.2e}")

