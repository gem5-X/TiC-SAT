ifeq ($(SA_SIZE),)
$(error SA_SIZE is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(MODEL),)
$(error MODEL is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(ACTIVATION_BITS),)
$(error ACTIVATION_BITS is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(WEIGHT_BITS),)
$(error WEIGHT_BITS is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(ACTIVATION_FP),)
$(error ACTIVATION_FP is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(WEIGHT_FP),)
$(error WEIGHT_FP is not defined. Please call the script correctly: 'make SA_SIZE=n MODEL=model ACTIVATION_BITS=act_bits WEIGHT_BITS=w_bits ACTIVATION_FP=is_act_fp WEIGHT_FP=is_w_fp'.)
endif

ifeq ($(ACTIVATION_FP), 1)
ACT_TYPE_TAG = fp
else ifeq ($(ACTIVATION_FP), 0)
ACT_TYPE_TAG = int
else
$(error ACTIVATION_FP is not binary.)
endif

ifeq ($(WEIGHT_FP), 1)
WEIGHT_TYPE_TAG = fp
else ifeq ($(WEIGHT_FP), 0)
WEIGHT_TYPE_TAG = int
else
$(error WEIGHT_FP is not binary.)
endif

QUANTIZATION_TAG = act_$(ACT_TYPE_TAG)$(ACTIVATION_BITS)_w_$(WEIGHT_TYPE_TAG)$(WEIGHT_BITS)

ifneq ($(SW),)
SW_TAG = _SW
SW_DEFINE = -DSW_MULT
endif

# [-DSA, -DSIMD] [-DSA_SIZE=16] [-DBWMA] [-DMODEL=libritrans|librispeech] [-DRELOAD_WEIGHT] [-DDEVELOP] [-DCORE_NUM=4]
DEFINES = -DSA -DSA_SIZE=$(SA_SIZE) -DMODEL=$(MODEL) -DACTIVATION_BITS=$(ACTIVATION_BITS) -DWEIGHT_BITS=$(WEIGHT_BITS) -DACTIVATION_FP=$(ACTIVATION_FP) -DWEIGHT_FP=$(WEIGHT_FP) -DBWMA -DCORE_NUM=1 $(SW_DEFINE)

ARM_CXX = aarch64-linux-gnu-g++
LIBS = -lstdc++fs
CFLAGS = -fopenmp -O2 -Wall $(DEFINES)

TOP_OBJ_DIR = obj

OBJ_DIR = $(TOP_OBJ_DIR)/$(MODEL)_$(SA_SIZE)_$(QUANTIZATION_TAG)$(SW_TAG)

SRCS_CPP = $(wildcard transformer_layers/*.cpp accelerator/*.cpp)
SRCS_CC = $(wildcard transformer_layers/*.cc accelerator/*.cc)

_OBJ = $(patsubst %.cpp,%.o,$(SRCS_CPP)) $(patsubst %.cc,%.o,$(SRCS_CC))
OBJ = $(patsubst %,$(OBJ_DIR)/%,$(_OBJ))

HEADER_DEPS = $(wildcard *.h transformer_layers/*.h accelerator/*.h)

TARGET = $(MODEL)_$(SA_SIZE)_$(QUANTIZATION_TAG)$(SW_TAG).exe

# TEST_TARGET = test_$(SA_SIZE)_$(QUANTIZATION_TAG)$(SW_TAG).exe

$(OBJ_DIR)/%.o: %.cc $(HEADER_DEPS)
	@mkdir -p $(@D)
	$(ARM_CXX) -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/%.o: %.cpp $(HEADER_DEPS)
	@mkdir -p $(@D)
	$(ARM_CXX) -c -o $@ $< $(CFLAGS)

all: sim-shared/$(TARGET) 
# test: sim-shared/$(TEST_TARGET)

sim-shared/$(TARGET): $(OBJ_DIR)/transformer.o $(OBJ)
	$(ARM_CXX) -o $@ $^ $(CFLAGS) $(LIBS)

# sim-shared/$(TEST_TARGET): $(OBJ_DIR)/mat_mult_test.o $(OBJ)
# 	$(ARM_CXX) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -rf $(OBJ_DIR)/* sim-shared/$(TARGET)

full_clean:
	rm -rf $(TOP_OBJ_DIR)/* sim-shared/*.exe
