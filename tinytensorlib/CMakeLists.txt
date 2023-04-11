cmake_minimum_required(VERSION 3.20)
project(TinyTensorLib)

set(CMAKE_CXX_STANDARD 14)

set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DSA -DSA_SIZE=216")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "./*"
        "unittest/ChatfieldF.cpp"
        "eigen/Eigen/*"
        "../accelerator/*"
        )

add_executable(TinyTensorLib unittest/ChatfieldF.cpp ${transformer_SRC})
