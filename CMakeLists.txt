cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 14)


file(GLOB transformer_SRC
        "transformer_layers/*.h"
        "transformer_layers/*.cc"
        "accelerator/smm_gem.cpp"
        )

add_executable(FvllMontiTransformer transformer.cpp ${transformer_SRC})
