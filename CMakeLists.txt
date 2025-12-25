cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(merlin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# Find Python + NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# pybind11 via FetchContent (works well in CLion)
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Build the Python extension module
pybind11_add_module(merlin MODULE
        src/python/bindings.cpp
        src/core/option_pricer.cu
        src/core/calibration/ssvi.cpp
)

# Install the extension into the top-level of the package
install(TARGETS merlin
        LIBRARY DESTINATION .          # Unix-like
        ARCHIVE DESTINATION .          # just in case
        RUNTIME DESTINATION .          # Windows .pyd/.dll
)

target_include_directories(merlin PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(merlin PRIVATE
        CUDA::cudart
)

# On some platforms ensure no "lib" prefix for Python module
set_target_properties(merlin PROPERTIES PREFIX "")