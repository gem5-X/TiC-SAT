/* Copyright EPFL 2023
 * Joshua Klein
 *
 * Layer definitions for Chatfield-F variant CNN based on VGG8 [1].
 *
 * [1] Chatfield, Ken, et al. "Return of the devil in the details: Delving deep
 * into convolutional nets." arXiv preprint arXiv:1405.3531 (2014).
 *
 */

#ifndef __CHATFIELDF_HH__
#define __CHATFIELDF_HH__

// Number of inferences.
const int T_x = 1;
int layer_num = 0;

// Layers.
conv_layer_args conv1 = conv_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        0,                  // Assigned thread.
        true,               // Is first layer.
        false,              // Is last layer.
        SINGLE_BUFFER_TYPE, // Buffer type.
        224,                // Input height.
        224,                // Input width.
        3,                  // Input channels.
        11,                 // Kernel height.
        11,                 // Kernel width.
        64,                 // Number of filters.
        4,                  // Stride.
#if defined (AIMC)
true,               // Are we using AIMC tiles?
-1,                 // Allocated tile height? (-1 = infinite)
-1,                 // Allocated tile width? (-1 = infinite)
#endif
0,                  // Padding.
LRN_NORM_TYPE,      // Normalization.
RELU_ACT_TYPE       // Activation function.
);

pool_layer_args pool1 = pool_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        conv1,              // Input layer.
        2,                  // Pool dimension.
        2,                  // Pool factor.
        1,                  // Stride.
        MAX_POOL_TYPE,      // Pooling type.
        NO_NORM_TYPE,       // Normalization.
        NO_ACT_TYPE         // Activation function.
        );

conv_layer_args conv2 = conv_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        pool1,              // Input layer.
        5,                  // Kernel height.
        5,                  // Kernel width.
        256,                // Number of filters.
        1,                  // Stride.
#if defined (AIMC)
true,               // Are we using AIMC tiles?
#endif
2,                  // Padding.
LRN_NORM_TYPE,      // Normalization.
RELU_ACT_TYPE       // Activation function.
);

pool_layer_args pool2 = pool_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        conv2,              // Input layer.
        2,                  // Pool dimension.
        2,                  // Pool factor.
        1,                  // Stride.
        MAX_POOL_TYPE,      // Pooling type.
        NO_NORM_TYPE,       // Normalization.
        NO_ACT_TYPE         // Activation function.
        );

conv_layer_args conv3 = conv_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        pool2,              // Input layer.
        3,                  // Kernel height.
        3,                  // Kernel width.
        256,                // Number of filters.
        1,                  // Stride.
#if defined (AIMC)
true,               // Are we using AIMC tiles?
#endif
1,                  // Padding.
NO_NORM_TYPE,       // Normalization.
RELU_ACT_TYPE       // Activation function.
);

conv_layer_args conv4 = conv_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        conv3,              // Input layer.
        3,                  // Kernel height.
        3,                  // Kernel width.
        256,                // Number of filters.
        1,                  // Stride.
#if defined (AIMC)
true,               // Are we using AIMC tiles?
#endif
1,                  // Padding.
NO_NORM_TYPE,       // Normalization.
RELU_ACT_TYPE       // Activation function.
);

conv_layer_args conv5 = conv_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        conv4,              // Input layer.
        3,                  // Kernel height.
        3,                  // Kernel width.
        256,                // Number of filters.
        1,                  // Stride.
#if defined (AIMC)
true,               // Are we using AIMC tiles?
#endif
1,                  // Padding.
NO_NORM_TYPE,       // Normalization.
RELU_ACT_TYPE       // Activation function.
);

pool_layer_args pool3 = pool_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        conv5,              // Input layer.
        2,                  // Pool dimension.
        2,                  // Pool factor.
        1,                  // Stride.
        MAX_POOL_TYPE,      // Pooling type.
        NO_NORM_TYPE,       // Normalization.
        NO_ACT_TYPE         // Activation function.
        );

flatten_layer_args flatten1 = flatten_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        pool3               // Input height.
        );

fc_layer_args dense1 = fc_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        flatten1,           // Input layer.
#if defined (AIMC)
false,              // Are we using AIMC tiles?
#endif
4096,               // Output size.
NO_NORM_TYPE,       // Normalization.
RELU_ACT_TYPE       // Activation function.
);

fc_layer_args dense2 = fc_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        SINGLE_BUFFER_TYPE, // Buffer type.
        dense1,             // Input layer.
#if defined (AIMC)
false,              // Are we using AIMC tiles?
#endif
4096,               // Output size.
NO_NORM_TYPE,       // Normalization.
RELU_ACT_TYPE       // Activation function.
);

fc_layer_args dense3 = fc_layer_args(
        layer_num++,        // Layer number.
        T_x,                // Number of inferences.
        0,                  // Assigned thread.
        false,              // Is first layer.
        true,               // Is last layer.
        SINGLE_BUFFER_TYPE, // Buffer type.
        dense2.output_size, // Input size.
#if defined (AIMC)
false,              // Are we using AIMC tiles?
-1,                 // Allocated tile height? (-1 = infinite)
-1,                 // Allocated tile width? (-1 = infinite)
#endif
1000,               // Output size.
NO_NORM_TYPE,       // Normalization.
SOFTMAX_ACT_TYPE    // Activation function.
);

#endif // __CHATFIELDF_HH__
