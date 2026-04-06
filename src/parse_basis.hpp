#include "basisset.hpp"
#include "molecule.hpp"
#include <string>

namespace NuKEXC {

GTOBasisSet<double> parse_basis( const Molecule& mol,
                              std::string     fname,
                              SphericalType   sph    );

}
