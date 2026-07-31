from spack.package import *


class OpenOrbitalOptimizer(CMakePackage):
    """OpenOrbitalOptimizer: SCF orbital optimization library used by
    NuKEXC's standalone_hf test/benchmark executable."""

    homepage = "https://github.com/susilehtola/OpenOrbitalOptimizer"
    git = "https://github.com/susilehtola/OpenOrbitalOptimizer.git"


    #version("master", branch="master")
    version("0.3.0", tag="v0.3.0")

    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")
    depends_on("armadillo")

    def cmake_args(self):
        return [
            self.define("OPENORBITALOPTIMIZER_ENABLE_TESTS", False),
        ]
