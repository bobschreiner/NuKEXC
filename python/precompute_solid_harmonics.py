#!/usr/bin/env python3
"""
Generate the precomputed real solid-harmonic switch functions used by
nukexc/spherical_harmonics.hpp.

Why this exists / what changed
------------------------------
The previous generator expanded every harmonic into a flat sum of monomials and
unrolled the powers (``expand`` + string ``replace_pow``).  For high angular
momentum that produces long single expressions such as

    val = c0*x*x*x*x*x*x + c1*x*x*x*x*y*y + ... (up to ~10 degree-7 terms)

where each monomial rematerialises ``x*x*x*...`` from scratch.  ``ncu`` shows the
integral kernels are *register limited* (Block Limit Registers is the binding
occupancy limit), and these high-l cases are the biggest contributor to the
live-register set while evaluating the basis functions.

This version runs SymPy common-subexpression elimination (``cse``) per switch
case, so shared monomials (``x*x``, ``x*y``, ``x2*z2`` ...) are computed once into
local ``const double`` temporaries and reused.  That shortens the dependency
chains and cuts the number of simultaneously-live temporaries in exactly the
cases that dominate register pressure.  For the value+gradient function the
``cse`` is run jointly over (val, dx, dy, dz) so the four components share
temporaries.

Two functions are emitted (drop-in replacements for the ones currently in
spherical_harmonics.hpp):

  * real_solid_harmonic_cart_precomputed           -- value only
        (this is the one on the hot integral path, via eval_solid_harmonic /
         basis_eval_fast)
  * real_solid_harmonic_cart_and_grad_precomputed  -- value + gradient

Numerics are unchanged: coefficients are still evaluated to 17 significant
digits, and cse only factors identical subexpressions, so the double-precision
result matches the previous generator.

Usage
-----
    python precompute_solid_harmonics.py > solid_harmonics_switch.inc

then replace the two corresponding functions in spherical_harmonics.hpp with the
generated bodies.
"""

from sympy import (binomial, cse, diff, expand, factorial, numbered_symbols, pi,
                   Rational, simplify, sqrt, symbols)
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.printing.c import ccode

x, y, z = symbols("x y z", real=True)

L_MAX = 8   # generate l = 0 .. L_MAX-1
PREC = 17   # significant digits for the emitted coefficients

# Rewrites integer powers x**k (k <= limit) into plain products x*x*...,
# so the emitted C never calls pow() for these small exponents.
_expand_pow = create_expand_pow_optimization(2 * L_MAX + 2)


# --- symbolic definition of the real solid harmonics ----------------------
def poly_P(r, z, l, m):
    result = 0
    for k in range((l - m) // 2 + 1):
        result += ((-1) ** k * Rational(1, 2 ** l)
                   * binomial(l, k) * binomial(2 * l - 2 * k, l)
                   * (factorial(l - 2 * k) / factorial(l - 2 * k - m))
                   * r ** (2 * k) * z ** (l - 2 * k - m))
    return sqrt(Rational(factorial(l - m), factorial(l + m))) * result


def poly_A(x, y, m):
    result = 0
    for p in range(m + 1):
        cos_val = 0
        if (m - p) % 2 == 0:
            cos_val = (-1) ** ((m - p) // 2)
        result += binomial(m, p) * x ** p * y ** (m - p) * cos_val
    return result


def poly_B(x, y, m):
    result = 0
    for p in range(m + 1):
        sin_val = 0
        if (m - p) % 2 != 0:
            sin_val = (-1) ** ((m - p - 1) // 2)
        result += binomial(m, p) * x ** p * y ** (m - p) * sin_val
    return result


def solid_harmonic(l, m, x, y, z):
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    abs_m = abs(m)
    P = poly_P(r, z, l, abs_m)
    if m == 0:
        return sqrt(Rational(2 * l + 1, 4) / pi) * P
    elif m > 0:
        return sqrt(Rational(2 * l + 1, 2) / pi) * P * poly_A(x, y, abs_m)
    else:
        return sqrt(Rational(2 * l + 1, 2) / pi) * P * poly_B(x, y, abs_m)


# --- C++ emission ---------------------------------------------------------
def _ccode(expr):
    """C code for a pow-free version of expr."""
    return ccode(_expand_pow(expr))


def _is_zero(expr):
    return expr == 0 or bool(getattr(expr, "is_zero", False))


def emit_case(idx, l, m, names_exprs):
    """Emit one switch case.

    names_exprs is a list of (lvalue_name, sympy_expr).  A single joint cse is
    run over all components so temporaries are shared across value and gradient.
    """
    names = [n for n, _ in names_exprs]
    exprs = [_expand_pow(expand(e).evalf(PREC)) for _, e in names_exprs]

    # numbered_symbols('t') -> t0, t1, ... (no clash with x, y, z)
    repl, reduced = cse(exprs, symbols=numbered_symbols("t"))

    print(f"  case {idx}: {{ // l={l}, m={m}")
    for sym, sub in repl:
        print(f"    const double {ccode(sym)} = {_ccode(sub)};")
    for name, e in zip(names, reduced):
        if not _is_zero(e):
            print(f"    {name} = {_ccode(e)};")
    print("    break;")
    print("  }")


# Precompute S and its gradient for every (l, m).
harmonics = {}
for l in range(L_MAX):
    for m in range(-l, l + 1):
        S = simplify(solid_harmonic(l, m, x, y, z))
        harmonics[(l, m)] = (S, diff(S, x), diff(S, y), diff(S, z))

# ---- value-only function -------------------------------------------------
print("KOKKOS_INLINE_FUNCTION")
print("void real_solid_harmonic_cart_precomputed(const int l, const int m,")
print("                                          const double x, const double y,")
print("                                          const double z, double &val) {")
print("")
print("  // Initialize outputs upfront to zero out registers and clean up branch code")
print("  val = 0.0;")
print("")
print("  const int idx = l * l + l + m;")
print("  switch (idx) {")
for (l, m), (S, _, _, _) in harmonics.items():
    emit_case(l * l + l + m, l, m, [("val", S)])
print("  default:")
print("    break;")
print("  }")
print("}")
print("")

# ---- value + gradient function ------------------------------------------
print("KOKKOS_INLINE_FUNCTION")
print("void real_solid_harmonic_cart_and_grad_precomputed(")
print("    const int l, const int m, const double x, const double y, const double z,")
print("    double &val, double &dx, double &dy, double &dz) {")
print("")
print("  // Initialize outputs upfront to zero out registers and clean up branch code")
print("  val = 0.0;")
print("  dx = 0.0;")
print("  dy = 0.0;")
print("  dz = 0.0;")
print("")
print("  const int idx = l * l + l + m;")
print("  switch (idx) {")
for (l, m), (S, dSdx, dSdy, dSdz) in harmonics.items():
    emit_case(l * l + l + m, l, m,
              [("val", S), ("dx", dSdx), ("dy", dSdy), ("dz", dSdz)])
print("  default:")
print("    break;")
print("  }")
print("}")
