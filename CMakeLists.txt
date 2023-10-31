cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 17)

set(GCC_COVERAGE_COMPILE_FLAGS "-DREARRANGE -DDEVELOP -DHIDDEN_FLAG -DSA -DSA_SIZE=16 -DRELOAD_WEIGHT")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "transformer_layers/*.h"
        "transformer_layers/*.cc"
        "accelerator/smm_gem.cpp"
        "accelerator/systolic_m2m.cc"
        )

add_executable(FvllMontiTransformer transformer.cpp ${transformer_SRC})
