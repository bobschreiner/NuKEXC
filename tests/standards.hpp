/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 * (edited) 2026 Bob Schreiner
 *
 * All rights reserved.
 *
 * See https://github.com/wavefunction91/GauXC LICENSE.txt for details
 */

#include "../src/basisset.hpp"
#include "../src/molecule.hpp"

namespace NuKEXC {

Molecule make_water();
Molecule make_benzene();
Molecule make_ubiquitin();
Molecule make_taxol();
GTOBasisSet<double> make_631Gd(const Molecule &, SphericalType);

} // namespace NuKEXC
