cmake_minimum_required(VERSION 3.8)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 17)

<<<<<<< Updated upstream
set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DSA_SIZE=4 -DACTIVATION_BITS=32 -DWEIGHT_BITS=32 -DACTIVATION_FP=1 -DWEIGHT_FP=1 -DMODEL=libritrans") #-DDEVELOP -DRELOAD_WEIGHT -DSW_MULT
=======
set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DSA_SIZE=4 -DACTIVATION_BITS=32 -DWEIGHT_BITS=32 -DACTIVATION_FP=1 -DWEIGHT_FP=1 -DMODEL=librispeech") #-DDEVELOP -DRELOAD_WEIGHT -DSW_MULT
>>>>>>> Stashed changes
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} -O3" )

file(GLOB transformer_HEADER
        "transformer_layers/*.h"
)

file(GLOB transformer_SRC
        "transformer_layers/*.cc"
        "accelerator/smm_gem.cpp"
        "accelerator/sparseMatrixMultiplication.cpp"
        "accelerator/systolic_m2m.cc"
        )

set_source_files_properties(mat_mult_test.cpp PROPERTIES COMPILE_FLAGS "-Wall")# -fanalyzer")

add_executable(FvllMontiTransformer transformer.cpp ${transformer_SRC} ${transformer_HEADER})
add_executable(MultTest mat_mult_test.cpp accelerator/systolic_m2m.cc accelerator/smm_gem.cpp ${transformer_HEADER})
