cmake_minimum_required(VERSION 3.20)
project(GemminiTransformer)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "transformer.c"
        "include/*.h"
        )

add_executable(GemminiTransformer ${transformer_SRC})