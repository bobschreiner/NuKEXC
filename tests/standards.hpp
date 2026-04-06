#include "../src/basisset.hpp"
#include "../src/molecule.hpp"

namespace NuKEXC {

Molecule make_water();
Molecule make_benzene();
Molecule make_ubiquitin();
Molecule make_taxol();
GTOBasisSet<double> make_631Gd(const Molecule &, SphericalType);

} // namespace NuKEXC
