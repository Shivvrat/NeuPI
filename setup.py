import os
import sys

import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Enable global Cython compiler directives for optimization
Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False



class BuildExt(build_ext):
    """Sets the optimization and OpenMP flags for the compiler in use.

    No -march=native or -ffast-math: wheels must run on any CPU and keep IEEE semantics.
    OpenMP only parallelizes the UAI parser's prange loops; without it they run serially.
    It is off on macOS, whose clang has no OpenMP without libomp, and can be turned off
    anywhere with NEUPI_OPENMP=0.
    """

    def build_extensions(self):
        use_openmp = os.environ.get("NEUPI_OPENMP", "0" if sys.platform == "darwin" else "1") != "0"
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/O2"] + (["/openmp"] if use_openmp else [])
            link_args = []
        else:
            compile_args = ["-O3"] + (["-fopenmp"] if use_openmp else [])
            link_args = ["-fopenmp"] if use_openmp else []
        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args
        super().build_extensions()


# Define a single list containing all Cython extension modules
extensions = [
    # Extension 1: The UAI Reader
    Extension(
        "neupi.training.pm_ssl.io.uai_reader_cython",
        ["neupi/training/pm_ssl/io/uai_reader_cython.pyx"],
        include_dirs=[
            numpy.get_include(),
            # If you use PyTorch C++ headers here, uncomment the next line
            # *torch.utils.cpp_extension.include_paths(),
        ],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    # Extension 2: The KNN Discretizer Helper
    Extension(
        "neupi.discretize.cython_kn.kn_binary_vectors",
        ["neupi/discretize/cython_kn/kn_binary_vectors.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Make a single call to setup() with the list of all extensions
setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
        annotate=True,  # Generates a helpful HTML report for optimization
    )
)
