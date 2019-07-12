cmake_minimum_required(VERSION 2.8.12)
project(kanapy)

add_subdirectory(libs/pybind11)                          # Add subdir for pybuind11

include_directories(libs/eigen_cpp)                      # Add eigen path
pybind11_add_module(base src/kanapy/overlapDetect.cpp)   # Create the package
