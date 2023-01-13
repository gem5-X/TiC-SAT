# TiC-SAT

TiC-SAT is an architecture and framework for tightly-coupled systolic arrays devoted to accelerating transformer applications. 
It is a  model for systolic array acceleration in the gem5-X full system simulator and defined its interface with custom extensions to the ARMv8 instruction set. 

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
Once you have installed gem5-x, you can follow Section 3.3 of [this document](gem5_X_TechnicalManual_TiCSAT.pdf) to install *diod* and build the gem5-x binary again. This feature enables us to mount a shared folder from the host system in the gem5-x.

## Compiling a transformer code
In this repository, you can find the code to simulate a transformer model. To compile the code, you can use compile command inside the gem5-x system; however, it is not recommended because it takes several hours. The alternative way is to compile the code on your host system using the following command, put the compiled code in the shared folder, and mount it in the gem5-x system.
``` script
arm-linux-gnueabi-g++ transformer.cpp -o transformer.o
```
Once you have generated the output file, move transformer.o to the shared folder. Mount the folder in your gem5-x simulation.
Now, you can run your code on gem5-x by the following command:
``` script
./transformer.o
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
3. Operation bit width

You can follow the instruction in Section 8 of [this document](gem5_X_TechnicalManual_TiCSAT.pdf) to customize the accelrator. After applying any modification in the files in gem5-X-TiC-SAT, don't forget to use *scons* to recompile the gem5-X-TiC-SAT binary with the new structure (Section 2.2.2).

Please note that to change the configuration of the accelerator, you also need to modify the configuration on the software in [this file](accelerator/smm_gem.cpp). For instance, the systolic array size in this file, should be identical to the size assigned in the gem5-X-TiC-SAT.

## Reference
If you have used TiC-SAT, we would appreciate it if you cite the following paper in your academic articles:

```
A. Amirshahi, J. Klein, G. Ansaloni, D. Atienza , "TiC-SAT: Tightly-coupled Systolic Accelerator for Transformers", 
28th Asia and South Pacific Design Automation Conference (ASPDAC '23), Tokyo, Japan, doi: 10.1145/3566097.3567867
```

## Acknowledgements
This work has been supported by the EC H2020 WiPLASH (GA No. 863337) and the EC H2020 FVLLMONTI (GA No. 101016776) projects.

