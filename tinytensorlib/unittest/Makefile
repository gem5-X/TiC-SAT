##########################################
# Makefile for tinytensorlib unit tests. #
# Copyright EPFL 2023                    #
# Joshua Klein                           #
##########################################

CC          = g++
CFLAGS      = -O3 -w -std=c++17 -fcompare-debug-second #-mfpu=neon
THREADS     = -pthread
# LIBS	= -lm -lrt -lpthread

# Dirs to include
AIMCF       = -I ../../aimclib
INC         = -I ./ -I ../../eigen -I ../

# Change if you're using AIMC checker here.
CHECKER     = -DUSE_CHECKER

# Adding a new NN?  Put its name here.
# NOTE: The base name in name.upper() must reflect the ifdefs of the models
# file and file names.  E.g., cnn --> cnn.cpp/hh --> compile with -DCNN -->
# get cnn_analog.out and cnn_digital.out.
TARGETS_LIST=partial_dense partial_conv pthread_conv pthread_dense single_conv
ANA_ONLY_LIST=dedicated_partial_dense dedicated_partial_conv

###################################
# Which benchmarks are we making? #
###################################

# Test case.
test:
	$(CC) $(CFLAGS) $(INC) $(THREADS) test.cpp -o test.out

# Compile both digital and analog variant of application.
$(TARGETS_LIST):
	$(CC) $(CFLAGS) $(INC) $(THREADS) -D$(shell echo '$@' | tr '[:lower:]' '[:upper:]') $@.cpp -o $@_digital.out
	$(CC) $(CFLAGS) $(INC) $(THREADS) -D$(shell echo '$@' | tr '[:lower:]' '[:upper:]') -DAIMC $(AIMCF) $(CHECKER) $@.cpp -o $@_analog.out

$(ANA_ONLY_LIST):
	$(CC) $(CFLAGS) $(INC) $(THREADS) -D$(shell echo '$@' | tr '[:lower:]' '[:upper:]') -DAIMC $(AIMCF) $(CHECKER) $@.cpp -o $@_analog.out

all: $(TARGETS_LIST)
	echo "Compiled ALL models."

clean:
	rm -f ./*.out
