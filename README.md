# TiC-SAT

TiC-SAT is an architecture and framework for tightly-coupled systolic arrays devoted to accelerating transformer applications. 
It is a  model for systolic array acceleration in the gem5-X full system simulator, and defined its interface with custom extensions to the ARMv8 instruction set. 

## Installing gem5-x
You can follow Section 2 of [this documnet](https://www.epfl.ch/labs/esl/wp-content/uploads/2021/08/gem5_X_TechnicalManual_v2.pdf) for installing gem5-x.

## Compiling a transformer code
Since gem5-x is an arm-based simulator, the compiler should target arm-linux system. You can use the following command:
``` script
arm-linux-gnueabi-g++ mat_mul.cpp -o mat_mul.o
```
Once you generated the output file, move it to a folder which called shared.

## Runing code on gem5-x
```
./build/ARM/gem5.fast \
--remote-gdb-port=0 \
-d ./output \
./configs/example/fs.py \
--cpu-clock=1GHz \
--kernel=./full_system_images/binaries/vmlinux-5.4.0 \
--machine-type=VExpress_GEM5_V1 \
--dtb-file=./system/arm/dt/armv8_gem5_v1_8cpu.dtb \
-n 8 \
--disk-image=./full_system_images/disks/test_spm.img \
--caches \
--l2cache \
--l1i_size=32kB \
--l1d_size=32kB \
--l2_size=512kB \
--l2_assoc=2 \
--mem-type=DDR4_2400_4x16 \
--mem-ranks=4 \
--mem-size=4GB \
--sys-clock=1600MHz \
--cpu-type=MinorCPU
```

In a new terminal:
``` script
telnet localhost 3456
```
Once the ubuntu system boots, you can use the following command to mount the shared folder in your gem5-x simulation.
```
bash mount.sh <absolute_path_to_you_shared_folder>
```
Now, you can run your code on gem5-x.

## Design your own TiC-SAT

### SA accelerator kernel size (k)
You can change this parameter in the following files to have your prefered systolic array size. 

**src/dev/arm/systolic_m2m.hh**

For instance, if you want an SA with size 8, you can modify the following code:
``` C++
#define KERNEL_DIM 8
#define MAX_COL 2
```

Don't forget to use *scons* when you modified the gem5-x kernel to compile it again with the new structure.

## Reference
If you have used TiC-SAT in your academic articles, we appreciate it if you cite the following paper:

```
A. Amirshahi *et al*, "TiC-SAT: Tightly-coupled Systolic Accelerator for Transformers", 
28th Asia and South Pacific Design Automation Conference (ASPDAC '23), Tokyo, Japan, doi: 10.1145/3566097.3567867
```



## Acknowledgements
This work has been supported by the EC H2020 WiPLASH (GA No. 863337) and the EC H2020 FVLLMONTI (GA No. 101016776) projects.

