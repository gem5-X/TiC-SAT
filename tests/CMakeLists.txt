cmake_minimum_required(VERSION 3.20)
project(FvllMontiMatrixMultiplication)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "../kernels/matmul.h",
        "../kernels/matmul.cc"
        )

add_executable(FvllMontiMatrixMultiplication matmulTest.cpp ${transformer_SRC})
