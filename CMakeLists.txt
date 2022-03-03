cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "kernels/*.h"
        "kernels/*.cc"
        "kernels/matMulSystolic.*"
        "kernels/matMulSystolic.h"
        "kernels/systolic_m2m.cpp"
        "gem5x_codes/smm_gem.cpp"
        )

add_executable(FvllMontiTransformer selfattentest.cpp ${transformer_SRC})
