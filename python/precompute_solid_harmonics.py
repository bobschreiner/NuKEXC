from sympy import *
from sympy.printing.c import ccode
x, y, z, r = symbols('x y z r', real=True)

def poly_P(r, z, l, m):
    """Your existing poly_P in SymPy"""
    result = 0
    for k in range((l - m)//2 + 1):
        result += ((-1)**k * Rational(1, 2**l) * 
                   binomial(l, k) * binomial(2*l - 2*k, l) *
                   (factorial(l - 2*k) / factorial(l - 2*k - m)) *
                   r**(2*k) * z**(l - 2*k - m))
    return sqrt(Rational(factorial(l-m), factorial(l+m))) * result

def poly_A(x, y, m):
    result = 0
    for p in range(m + 1):
        cos_val = 0
        if (m - p) % 2 == 0:
            cos_val = (-1)**((m-p)//2)
        result += binomial(m, p) * x**p * y**(m-p) * cos_val
    return result

def poly_B(x, y, m):
    result = 0
    for p in range(m + 1):
        sin_val = 0
        if (m - p) % 2 != 0:
            sin_val = (-1)**((m-p-1)//2)
        result += binomial(m, p) * x**p * y**(m-p) * sin_val
    return result

def solid_harmonic(l, m, x, y, z):
    r2 = x**2 + y**2 + z**2
    r = sqrt(r2)
    abs_m = abs(m)
    P = poly_P(r, z, l, abs_m)
    if m == 0:
        return sqrt(Rational(2*l+1, 4) / pi) * P
    elif m > 0:
        return sqrt(Rational(2*l+1, 2) / pi) * P * poly_A(x, y, abs_m)
    else:
        return sqrt(Rational(2*l+1, 2) / pi) * P * poly_B(x, y, abs_m)

def generate_gradient(l, m):
    S = solid_harmonic(l, m, x, y, z)
    S = simplify(S)
    
    dSdx = simplify(diff(S, x))
    dSdy = simplify(diff(S, y))
    dSdz = simplify(diff(S, z))
    
    return dSdx, dSdy, dSdz

def generate_cpp_case(l, m):
    S = simplify(solid_harmonic(l, m, x, y, z))
    dSdx, dSdy, dSdz = generate_gradient(l, m)
    
   # print(f"// l={l}, m={m}")
    print(f"case {m}:")
    print(f"    val = {ccode(S)};")
    print(f"    return;")

   

# 1. Generate all harmonics expressions
harmonics = {}
l_max = 8 
for l in range(l_max):
    for m in range(-l, l+1):
        # Substitute your package's solid_harmonic function here
        S = simplify(solid_harmonic(l, m, x, y, z))
        dSdx = simplify(diff(S, x))
        dSdy = simplify(diff(S, y))
        dSdz = simplify(diff(S, z))
        harmonics[(l,m)] = (S, dSdx, dSdy, dSdz)

# Your custom string-replacement function for unrolling powers
def replace_pow(S, k_max):
    for k in range(2, k_max):
        for s in ["x", "y", "z"]:
            S = S.replace(f"pow({s}, {k})", f"{s}*" * (k - 1) + f"{s}")
    return S

# 2. Print out the structured Kokkos C++ Code
print("KOKKOS_INLINE_FUNCTION")
print("void real_solid_harmonic_cart_and_grad_precomputed(")
print("    const int l, const int m, ")
print("    const double x, const double y, const double z,")
print("    double &val, double &dx, double &dy, double &dz) {")
print("")
print("  // Initialize outputs upfront to zero out registers and clean up branch code")
print("  val = 0.0; dx = 0.0; dy = 0.0; dz = 0.0;")
print("")
print("  const int idx = l * l + l + m;")
print("  switch (idx) {")

for (l, m), (S, gx, gy, gz) in harmonics.items():
    max_power = 2 * l_max + 1
    
    # Evaluate constants to 64-bit precision decimals (.evalf(17)) and unroll groupings
    Sc  = replace_pow(ccode(expand(S).evalf(17)), max_power)
    gxc = replace_pow(ccode(expand(gx).evalf(17)), max_power)
    gyc = replace_pow(ccode(expand(gy).evalf(17)), max_power)
    gzc = replace_pow(ccode(expand(gz).evalf(17)), max_power)
    
    lm_index = l * l + l + m
    print(f"  case {lm_index}: // l={l}, m={m}") 
    
    # Only print assignments if they are not explicitly zero to minimize compiled instructions
    if Sc  != "0" and Sc  != "0.0": print(f"    val = {Sc};")
    if gxc != "0" and gxc != "0.0": print(f"    dx  = {gxc};")
    if gyc != "0" and gyc != "0.0": print(f"    dy  = {gyc};")
    if gzc != "0" and gzc != "0.0": print(f"    dz  = {gzc};")
    
    print("    break;")

print("  default:")
print("    break;")
print("  }")
print("}")
