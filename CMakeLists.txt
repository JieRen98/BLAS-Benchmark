project(BLAS-Benchmark)
enable_language(CXX)
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(GFLAGS REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC Eigen3::Eigen BLAS::BLAS gflags)
target_compile_options(main PUBLIC -O3)
