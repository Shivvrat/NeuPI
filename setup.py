import numpy
import torch
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# Enable global Cython compiler directives for optimization from your original file
Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.generate_cleanup_code = False

# Define compiler and linker flags for optimization
extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]
extra_link_args = ["-fopenmp"]

# Define the Cython extension module.
# We specify the exact path to the .pyx file and its corresponding module path.
extensions = [
    Extension(
        "neupi.training.pm_ssl.io.uai_reader_cython",
        ["neupi/training/pm_ssl/io/uai_reader_cython.pyx"],
        include_dirs=[
            numpy.get_include(),
            # Use the modern, public PyTorch API to get C++ include paths
            # *torch.utils.cpp_extension.include_paths(),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# Use the setup function to configure the build process.
# setuptools will find this and use it to compile the extension.
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
