cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "kernels/*.h"
        "kernels/*.cc"
        )

add_executable(FvllMontiTransformer selfattentest.cpp ${transformer_SRC})
