/* 
 * Copyright EPFL 2023
 * Joshua Klein
 * 
 * This file is for organizing the includes necessary for running the Hybrid
 * CNNs.  Used primarily to keep main code slightly cleaner.
 *
 */

#ifndef __TINYTENSORLIB_HH__
#define __TINYTENSORLIB_HH__

// Eigen library stuff.
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/Dense"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
using namespace Eigen;
typedef Tensor<int8_t, 1> TB_Vector;
typedef Tensor<int8_t, 2> TB_Matrix2D;
typedef Tensor<int8_t, 3> TB_Matrix3D;

// AIMC tile helper functions and aimclib.
#if defined (AIMC)
#include "aimc.hh"
#include "aimc_utilities.hh"
#endif

// Main layer, function, utilty definitions.
#include "layer.hh"
#include "functions.hh"

// Special-built AIMC tile-enabled layers and functions.
#if defined (AIMC)
#include "aimc_layer.hh"
#include "aimc_functions.hh"
#endif

#include "threads.hh"
#include "utilities.hh"
#include "blocks.hh"

#endif // __TINYTENSORLIB_HH__
