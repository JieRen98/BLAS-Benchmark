project(BLAS-Benchmark)
enable_language(CXX)
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

option(USE_MKL "Use MKL or OpenBlas as backend" ON)

find_package(Eigen3 REQUIRED)
find_package(GFLAGS REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC Eigen3::Eigen gflags)

if (USE_MKL STREQUAL "ON")
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(main PUBLIC MKL::MKL)
    target_compile_options(main PUBLIC -DUSE_MKL=1)
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(main PUBLIC BLAS::BLAS)
    target_compile_options(main PUBLIC -DUSE_MKL=0)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    target_compile_options(main PUBLIC -O0)
else()
    target_compile_options(main PUBLIC -O3)
endif ()
