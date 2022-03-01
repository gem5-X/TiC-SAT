cmake_minimum_required(VERSION 3.20)
project(FvllMontiMatrixMultiplication)

set(CMAKE_CXX_STANDARD 14)


#file(GLOB transformer_SRC
#        "../kernels/matmul.cc",
#        "../kernels/matmul.h"
#        )

file(GLOB transformer_SRC
        "../kernels/systolic_m2m.h",
        "../kernels/systolic_m2m.cpp"
        "../kernels/matMulSystolic.cc"
        "../kernels/matMulSystolic.h"
        )

#add_executable(FvllMontiMatrixMultiplication matmulTest.cpp output_8bit.cpp ${transformer_SRC})
add_executable(FvllMontiMatrixMultiplication systolicTest.cpp ../gem5x_codes/full_weights.cpp ${transformer_SRC})
