# TiC-SAT

TiC-SAT is an architecture and framework for tightly-coupled systolic arrays devoted to accelerating transformer applications. 
It is a model for systolic array acceleration in the gem5-X full system simulator and defined its interface with custom extensions to the ARMv8 instruction set.

## Installing gem5-x
You can follow Section 2 of [this document](gem5_X_TechnicalManual_TiCSAT.pdf) for installing gem5-x. The installation steps are as follows. Please refer to the document for more details.
1. Register on the [gem5-X website](http://esl.epfl.ch/gem5-x), download the *full system files*, and set up the downloaded file's path in gem5-X-TiC-SAT of this repository.
2. Generate the device tree binary files
3. Build an ARM gem5 binary (gem5.fast is recommended)
4. Run your full system (FS) simulation

Running the FS simulation takes several minutes. Then, you should be able to access the kernel dmesg, followed by a login and a terminal in the gem5-X FS mode. You can use standard Linux commands in the terminal. You can exit the terminal using the following command in the connected terminal:
``` script
m5 exit
```

## Sharing files between gem5-x and the host systems
Once you have installed gem5-x, you can follow Section 3.3 of [this document](gem5_X_TechnicalManual_TiCSAT.pdf) to install *diod* and build the gem5-x binary again. This feature enables us to mount a shared folder from the host system in the gem5-x. The provided `Makefile` outputs the executable into the folder `sim-shared`, which can be mounted in gem5-x.

## Compiling a transformer code
In this repository, you can find the code to simulate a transformer model. To compile the code, you have two options. While using the compile command inside the gem5-x system is available, it is not recommended due to its prolonged execution time. Instead, follow the steps below to compile the code on your host system, place the compiled code in the shared folder, and then mount it in the gem5-x system.

This version allows the user to define the datatype of the weights and activations employed at the transformer and the systolic array. The previous INT8-only version can be found at the branch `int8_sa`.

Use the following compilation command on your host system:
``` script
make all
```
The parameters you can modify in the `Makefile` are:

- **-DSA** or **-DSIMD**: Indicates whether the system has a systolic array or an activated SIMD accelerator.
- **-DSA_SIZE**: Assigns the size of the systolic array, e.g., 16 for SA16x16 or 8 for 8x8.
- **-MODEL**: Chooses the architecture transformer model executed. This parameter allows `test_model | libritrans | librispeech`.
- **-ACTIVATION_BITS**: Sets the number of bits employed for the activations, e.g., 32 for FP32 or INT32.
- **-ACTIVATION_FP**: Defines if the format of the activation is floating-point (`1`) or integer (`0`).
- **-WEIGHT_BITS**: Sets the number of bits employed for the weights, e.g., 32 for FP32 or INT32.
- **-WEIGHT_FP**: Defines if the format of the weight is floating-point (`1`) or integer (`0`).
- **-DRELOAD_WEIGHT**: Reloads weights and input data from memory to ensure consistent data for experiments. Avoid using it if you are compiling the code for the first time. You need to modify the save directory to the `transformer.cpp` as `std::string dir_name = "/path/to/weight/directory"`.
- **-DDEVELOP**: Enables all develop/debug functions. This model does NOT use accelerators and is solely for debugging functions.
- **-DCORE_NUM**: Specifies the number of cores equipped with systolic array accelerators. For a single-core system, set it to 1. Dual- and quad-core systems have been tested.

The `Makefile` will create the output executable in `sim-shared/`. For correct execution, create the folder `sim-shared/data` before simulation. Mount the shared folder `sim-shared` in your gem5-x simulation.
Now, you can run your code on gem5-x by the following command:
``` script
./<MODEL>_<SA_SIZE>_act_<ACTIVATION_DATATYPE>_w_<WEIGHT_DATATYPE>.exe <SPARSITY_QVK> <SPARSITY_CONDENSE> <SPARSITY_FF0> <SPARSITY_FF1>
```
For example, to simulate the execution with a 4x4 systolic array employing FP32 activations and INT8 weights with no sparsity:
```
./test_model_4_act_fp32_w_int8.exe 0 0 0 0
```

## Extract the statistics
To extract more than 1000 timing and memory statistics from a piece of your code, you can add the following codes before and after the target code. The stats file will be created in the output directory.
``` C++
system("m5 resetstats");
// Put your code here
system("m5 dumpresetstats");
```
For instance, in this repository, we extract the statistics of each layer of the transformer using the abovementioned commands in [this file](transformer_layers/transformerBlock.cc).

## Advanced: Design your own TiC-SAT
You can change the following parameters in the gem5-X-TiC-SAT to customize TiC-SAT for your application:
1. Operation latency of custom instructions
2. Systolic array size
3. Data type of the weights and activations

You can follow the instructions in Section 8 of [this document](gem5_X_TechnicalManual_TiCSAT.pdf) to customize the accelerator. After applying any modification in the files in gem5-X-TiC-SAT, don't forget to use *scons* to recompile the gem5-X-TiC-SAT binary with the new structure (Section 2.2.2).

Please note that to change the configuration of the accelerator, you also need to modify the configuration on the software in [this file](transformer_layers/util.h). For instance, the systolic array size in this file, should be identical to the size assigned in the gem5-X-TiC-SAT.

## Reference
If you have used TiC-SAT, we would appreciate it if you cite the following papers in your academic articles:

```
P. Palacios, R. Medina, JL. Rouas, G. Ansaloni, D. Atienza, "Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems",
Great Lakes Symposium on VLSI (GLSVLSI) 2025, New Orleans, USA, doi: 10.1145/3716368.3735158

A. Amirshahi, J. Klein, G. Ansaloni, D. Atienza, "TiC-SAT: Tightly-coupled Systolic Accelerator for Transformers", 
28th Asia and South Pacific Design Automation Conference (ASP-DAC '23), Tokyo, Japan, doi: 10.1145/3566097.3567867

A. Amirshahi, G. Ansaloni, and D. Atienza, "Accelerator-driven Data Arrangement to Minimize Transformers Run-time on Multi-core Architectures",
arXiv preprint arXiv:2312.13000 (2023).
```

## Acknowledgements
This work has been supported by the EC H2020 WiPLASH Project (GA No. 863337), the EC H2020 FVLLMONTI Project (GA No. 101016776), the SwissChips Research Project, and the Swiss NSF Edge-Companions Project (GA No. 10002812). This research was partially conducted by ACCESS – AI Chip Center for Emerging Smart Systems, supported by the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government.

