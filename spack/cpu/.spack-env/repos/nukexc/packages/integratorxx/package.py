from spack.package import *


class Integratorxx(CMakePackage):
    """IntegratorXX: quadrature rules for numerical integration of atomic
    orbitals / molecular grids (radial + angular quadratures)."""

    homepage = "https://github.com/wavefunction91/IntegratorXX"
    git = "https://github.com/wavefunction91/IntegratorXX.git"


    # Pin to a real tag/commit once you know which one NuKEXC needs.
    version("master", branch="master")
    # version("0.1.0", tag="v0.1.0")

    variant("shared", default=True, description="Build shared libraries")

    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")

    def cmake_args(self):
        args = [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            # IntegratorXX has its own test suite driven by a bundled
            # FetchContent'd Catch2 - disable it when building as a
            # dependency to avoid pulling in unrelated network fetches.
            self.define("INTEGRATORXX_ENABLE_TESTS", False),
        ]
        return args
