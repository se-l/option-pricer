cmake_minimum_required(VERSION 4.1.2)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(merlin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# 1. Create a CORE library so we don't compile the same code twice
# We use OBJECT to make it easy to link into both a Python Module and a Shared Library
add_library(merlin_core OBJECT
        src/core/pricing_engine.cu
        src/core/pricing_engine.cpp
        src/core/calibration/ssvi.cpp
)
target_link_libraries(merlin_core PRIVATE CUDA::cudart)
# This handles the API export/import logic for the shared library build
target_compile_definitions(merlin_core PUBLIC MERLIN_EXPORTS)

# 2. Build the Python extension module
pybind11_add_module(merlin MODULE
        bindings/python/bindings.cpp
)
target_link_libraries(merlin PRIVATE merlin_core CUDA::cudart)
target_include_directories(merlin PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)
set_target_properties(merlin PROPERTIES PREFIX "")

# 3. Build the Shared Library for C# / .NET
add_library(merlin_native SHARED
        bindings/csharp/merlin_c_api.h
        bindings/csharp/merlin_c_api.cpp
)
# This is the crucial part: tell CMake to handle CUDA linking for this target
#set_target_properties(merlin_native PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(merlin_native PRIVATE merlin_core CUDA::cudart)

# 4. Install
install(TARGETS merlin DESTINATION .)
install(TARGETS merlin_native RUNTIME DESTINATION bin LIBRARY DESTINATION lib)

# Enable testing logic
enable_testing()

# Only add the tests directory if we are building the main project
# (prevents tests from being built if another project includes merlin)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(tests)
endif()