from setuptools import setup, Extension
import pybind11
from glob import glob

# Get all source files
core_sources = glob('src/core/*.cu') + glob('src/core/*.cpp')
python_sources = glob('src/python/*.cpp')

ext_modules = [
    Extension(
        'berlin',
        sources=core_sources + python_sources,
        include_dirs=[
            'include',
            'src',
            pybind11.get_include(),
            pybind11.get_include(True),
            '/usr/local/cuda/include'  # Adjust based on your CUDA installation
        ],
        library_dirs=['/usr/local/cuda/lib64'],
        libraries=['cudart'],
        extra_compile_args=['-std=c++14', '-O3'],
        extra_link_args=[],
        language='c++'
    ),
]

setup(
    name='berlin',
    version='0.1',
    ext_modules=ext_modules,
    zip_safe=False,
)