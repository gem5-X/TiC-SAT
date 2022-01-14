cmake_minimum_required(VERSION 3.20)
project(FvllMontiMatrixMultiplication)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "../kernels/systolic_m2m.h",
        "../kernels/systolic_m2m.cpp"
        )

add_executable(FvllMontiMatrixMultiplication systolicTest.cpp ${transformer_SRC})
