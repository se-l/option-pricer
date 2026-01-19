cmake_minimum_required(VERSION 4.1.2)
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Create the test executable
add_executable(merlin_tests
        core/test_pricing_engine.cpp
)

# Link against your core library (the OBJECT or STATIC library we created earlier)
target_link_libraries(merlin_tests PRIVATE
        Catch2::Catch2WithMain
        merlin_core
        CUDA::cudart
)

# Add the test to the CTest runner
add_test(NAME AllConsistencyTests COMMAND merlin_tests)