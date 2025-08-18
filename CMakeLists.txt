cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(gpu_module LANGUAGES CXX CUDA)

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
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Build the Python extension module
pybind11_add_module(gpu_module MODULE
    src/gpu_module.cu
)

target_include_directories(gpu_module PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(gpu_module PRIVATE
    CUDA::cudart
)

# On some platforms ensure no "lib" prefix for Python module
set_target_properties(gpu_module PROPERTIES PREFIX "")