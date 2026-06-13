from sympy import (symbols, sqrt, exp, pi, Rational, Integer,
                   factorial, binomial, diff, expand, simplify, S)
from sympy.printing.c import ccode

x, y, z, r, zeta = symbols('x y z r zeta', real=True, positive=True)

# ── Solid harmonic helpers ────────────────────────────────────────────────────
def poly_P(r_sym, z_sym, l, m):
    result = S.Zero
    for k in range((l - m) // 2 + 1):
        result += (S.NegativeOne**k * Rational(1, 2**l) *
                   binomial(l, k) * binomial(2*l - 2*k, l) *
                   Rational(factorial(l - 2*k), factorial(l - 2*k - m)) *
                   r_sym**(2*k) * z_sym**(l - 2*k - m))
    return sqrt(Rational(factorial(l - m), factorial(l + m))) * result

def poly_A(x_sym, y_sym, m):
    result = S.Zero
    for p in range(m + 1):
        cos_val = S.Zero
        if (m - p) % 2 == 0:
            cos_val = S.NegativeOne**((m - p) // 2)
        result += binomial(m, p) * x_sym**p * y_sym**(m - p) * cos_val
    return result

def poly_B(x_sym, y_sym, m):
    result = S.Zero
    for p in range(m + 1):
        sin_val = S.Zero
        if (m - p) % 2 != 0:
            sin_val = S.NegativeOne**((m - p - 1) // 2)
        result += binomial(m, p) * x_sym**p * y_sym**(m - p) * sin_val
    return result

def solid_harmonic(l, m, x_s, y_s, z_s):
    r2    = x_s**2 + y_s**2 + z_s**2
    r_sym = sqrt(r2)
    abs_m = abs(m)
    P = poly_P(r_sym, z_s, l, abs_m)
    if m == 0:
        return sqrt(Rational(2*l + 1, 4) / pi) * P
    elif m > 0:
        return sqrt(Rational(2*l + 1, 2) / pi) * P * poly_A(x_s, y_s, abs_m)
    else:
        return sqrt(Rational(2*l + 1, 2) / pi) * P * poly_B(x_s, y_s, abs_m)

# ── Incomplete gamma for positive integer n ───────────────────────────────────
def lower_gamma_int(n, x_sym):
    poly = sum(x_sym**k / factorial(k) for k in range(n))
    return factorial(n - 1) * (1 - exp(-x_sym) * poly)

def upper_gamma_int(n, x_sym):
    poly = sum(x_sym**k / factorial(k) for k in range(n))
    return factorial(n - 1) * exp(-x_sym) * poly

# ── Radial factor and prefactor ───────────────────────────────────────────────
def C_prefactor(n, l):
    return (4 * pi * (2 * zeta)**Rational(2*n + 1, 2)
            / (sqrt(factorial(2*n)) * (2*l + 1)))

def I_tilde(n, l):
    a  = n + l + 2
    b  = n - l + 1
    lg = lower_gamma_int(a, zeta * r)
    ug = upper_gamma_int(b, zeta * r)
    return lg / (zeta**a * r**(2*l + 1)) + ug / zeta**b

# ── Full potential ────────────────────────────────────────────────────────────
def compute_V(n, l, m):
    return C_prefactor(n, l) * simplify(solid_harmonic(l, m, x, y, z)) * I_tilde(n, l)

# ── Code-generation helpers ───────────────────────────────────────────────────
def make_cases(n_max):
    cases = []
    idx = 0
    for n in range(1, n_max + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                cases.append((idx, n, l, m))
                idx += 1
    return cases

# ── Emit C++ ──────────────────────────────────────────────────────────────────
n_max  = 3;
cases  = make_cases(n_max)
k_max  = 2 * n_max + 3

print("KOKKOS_INLINE_FUNCTION")
print("double sto_potential_pre(")
print("    const int    idx,")
print("    const double x, const double y, const double z,")
print("    const double r, const double zeta) {")
print()
print("  switch (idx) {")

for (idx, n, l, m) in cases:
    V  = compute_V(n, l, m)
    Vc = ccode(expand(V).evalf(17))

    print(f"  case {idx}: // n={n}, l={l}, m={m}")
    if Vc not in ("0", "0.0"):
        print(f"    return {Vc};")
    else:
        print(f"    return 0.0;")

print("  default: return 0.0;")
print("  }")
print("}")

C1 = C_prefactor(1,0).evalf(25)
I1 = I_tilde(1,0).evalf(25)
S1 = simplify(solid_harmonic(0, 0, x, y, z)).evalf(25)

print(f"C1 : {C1}")
print(f"I1 : {I1}")
print(f"S1 : {S1}")



