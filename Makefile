# [-DSA, -DSIMD] [-DSA_SIZE=16] [-DBWMA] [-DRELOAD_WEIGHT] [-DDEVELOP] [-DCORE_NUM=4]
DEFINES = -DSA -DSA_SIZE=16 -DBWMA -DCORE_NUM=1

ARM_CXX = aarch64-linux-gnu-g++
LIBS = 
CFLAGS = -fopenmp -O2 -Wall $(DEFINES)

OBJ_DIR = obj

_OBJ = transformer.o transformer_layers/addNorm.o transformer_layers/debuggerFunctions.o transformer_layers/dense.o transformer_layers/selfattention.o transformer_layers/softmax.o transformer_layers/transformerBlock.o transformer_layers/transpose.o accelerator/smm_gem.o accelerator/systolic_m2m.o
OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_OBJ))

HEADER_DEPS = transformer.h transformer_layers/addNorm.h transformer_layers/debuggerFunctions.h transformer_layers/dense.h transformer_layers/selfattention.h transformer_layers/softmax.h transformer_layers/transformerBlock.h transformer_layers/transpose.h transformer_layers/util.h accelerator/smm_gem.h accelerator/systolic_m2m.h

$(OBJ_DIR)/%.o: %.cc $(HEADER_DEPS)
	@mkdir -p $(@D)
	$(ARM_CXX) -c -o $@ $< $(CFLAGS)

all: sim-shared/transformer

sim-shared/transformer: $(OBJ)
	$(ARM_CXX) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -rf $(OBJ_DIR)/* sim-shared/transformer
