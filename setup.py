import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# Enable global Cython compiler directives for optimization
Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False

# Define compiler and linker flags for optimization
extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]
extra_link_args = ["-fopenmp"]

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
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    # Extension 2: The KNN Discretizer Helper
    Extension(
        "neupi.discretize.cython_kn.kn_binary_vectors",
        ["neupi/discretize/cython_kn/kn_binary_vectors.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Make a single call to setup() with the list of all extensions
setup(
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
