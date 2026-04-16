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

#include "basisset.hpp"
#include "molecule.hpp"
#include <string>

namespace NuKEXC {

GTOBasisSet<double> parse_basis( const Molecule& mol,
                              std::string     fname,
                              SphericalType   sph    );

}
