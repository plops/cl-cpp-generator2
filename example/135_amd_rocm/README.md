# Programs

- on gentoo (state August 2023)

## rocm-smi
```
======================= ROCm System Management Interface =======================
========================= Version of System Component ==========================
Driver version: 6.3.12-gentoo-x86_64
================================================================================
====================================== ID ======================================
GPU[0]		: GPU ID: 0x15e7
================================================================================
================================== Unique ID ===================================
GPU[0]		: Unique ID: N/A
================================================================================
==================================== VBIOS =====================================
GPU[0]		: VBIOS version: 113-BARCELO-003
================================================================================
================================= Temperature ==================================
GPU[0]		: Temperature (Sensor edge) (C): 54.0
================================================================================
========================== Current clock frequencies ===========================
GPU[0]		: fclk clock level: 2: (1200Mhz)
GPU[0]		: mclk clock level: 2: (1200Mhz)
GPU[0]		: sclk clock level: 1: (400Mhz)
GPU[0]		: socclk clock level: 0: (400Mhz)
================================================================================
============================== Current Fan Metric ==============================
GPU[0]		: Unable to detect fan speed for GPU 0
================================================================================
============================ Show Performance Level ============================
GPU[0]		: Performance Level: auto
================================================================================
=============================== OverDrive Level ================================
GPU[0]		: GPU OverDrive value (%): 0
================================================================================
=============================== OverDrive Level ================================
GPU[0]		: GPU Memory OverDrive value (%): 0
================================================================================
================================== Power Cap ===================================
GPU[0]		: Not supported on the given system
GPU[0]		: Max Graphics Package Power Unsupported
================================================================================
============================= Show Power Profiles ==============================
GPU[0]		: 1. Available power profile (#1 of 7): CUSTOM
GPU[0]		: 2. Available power profile (#2 of 7): VIDEO*
GPU[0]		: 3. Available power profile (#4 of 7): COMPUTE
GPU[0]		: 4. Available power profile (#5 of 7): VR
================================================================================
============================== Power Consumption ===============================
Not supported on the given system
GPU[0]		: Average Graphics Package Power (W): 5.0
================================================================================
========================= Supported clock frequencies ==========================
GPU[0]		: Supported dcefclk frequencies on GPU0
GPU[0]		: 0: 400Mhz
GPU[0]		: 1: 464Mhz
GPU[0]		: 2: 514Mhz
GPU[0]		: 3: 576Mhz
GPU[0]		: 4: 626Mhz *
GPU[0]		: 5: 685Mhz
GPU[0]		: 6: 757Mhz
GPU[0]		: 7: 847Mhz
GPU[0]		: 
GPU[0]		: Supported fclk frequencies on GPU0
GPU[0]		: 0: 400Mhz
GPU[0]		: 1: 800Mhz
GPU[0]		: 2: 1200Mhz *
GPU[0]		: 3: 1333Mhz
GPU[0]		: 
GPU[0]		: Supported mclk frequencies on GPU0
GPU[0]		: 0: 400Mhz
GPU[0]		: 1: 800Mhz
GPU[0]		: 2: 1200Mhz *
GPU[0]		: 3: 1333Mhz
GPU[0]		: 
GPU[0]		: Supported sclk frequencies on GPU0
GPU[0]		: 0: 200Mhz
GPU[0]		: 1: 400Mhz *
GPU[0]		: 2: 1800Mhz
GPU[0]		: 
GPU[0]		: Supported socclk frequencies on GPU0
GPU[0]		: 0: 400Mhz *
GPU[0]		: 1: 445Mhz
GPU[0]		: 2: 520Mhz
GPU[0]		: 3: 600Mhz
GPU[0]		: 4: 678Mhz
GPU[0]		: 5: 780Mhz
GPU[0]		: 6: 866Mhz
GPU[0]		: 7: 975Mhz
GPU[0]		: 
--------------------------------------------------------------------------------
================================================================================
============================== % time GPU is busy ==============================
GPU[0]		: GPU use (%): 0
================================================================================
============================== Current Memory Use ==============================
GPU[0]		: Not supported on the given system
GPU[0]		: Memory Activity: N/A
================================================================================
================================ Memory Vendor =================================
GPU[0]		: GPU memory vendor: unknown
================================================================================
============================= PCIe Replay Counter ==============================
GPU[0]		: PCIe Replay Count: 0
================================================================================
================================ Serial Number =================================
GPU[0]		: Serial Number: N/A
================================================================================
================================ KFD Processes =================================
No KFD PIDs currently running
================================================================================
============================= GPUs Indexed by PID ==============================
No KFD PIDs currently running
================================================================================
================== GPU Memory clock frequencies and voltages ===================
GPU[0]		: Requested function is not implemented on this setup
================================================================================
=============================== Current voltage ================================
GPU[0]		: Voltage (mV): 1306
================================================================================
================================== PCI Bus ID ==================================
GPU[0]		: PCI Bus: 0000:04:00.0
================================================================================
============================= Firmware Information =============================
GPU[0]		: ASD firmware version: 	0x2100009f
GPU[0]		: CE firmware version: 		79
GPU[0]		: DMCU firmware version: 	0
GPU[0]		: MC firmware version: 		0
GPU[0]		: ME firmware version: 		166
GPU[0]		: MEC firmware version: 	464
GPU[0]		: MEC2 firmware version: 	464
GPU[0]		: PFP firmware version: 	194
GPU[0]		: RLC firmware version: 	60
GPU[0]		: RLC SRLC firmware version: 	1
GPU[0]		: RLC SRLG firmware version: 	1
GPU[0]		: RLC SRLS firmware version: 	1
GPU[0]		: SDMA firmware version: 	40
GPU[0]		: SDMA2 firmware version: 	0
GPU[0]		: SMC firmware version: 	00.64.64.00
GPU[0]		: SOS firmware version: 	0x00000000
GPU[0]		: TA RAS firmware version: 	00.00.00.00
GPU[0]		: TA XGMI firmware version: 	00.00.00.00
GPU[0]		: UVD firmware version: 	0x00000000
GPU[0]		: VCE firmware version: 	0x00000000
GPU[0]		: VCN firmware version: 	0x06114000
================================================================================
================================= Product Info =================================
GPU[0]		: Card series: 		Barcelo
GPU[0]		: Card model: 		Radeon R9 M275
GPU[0]		: Card vendor: 		Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]		: Card SKU: 		BARCEL
================================================================================
================================== Pages Info ==================================
GPU[0]		: Not supported on the given system
============================ Show Valid sclk Range =============================
GPU[0]		: Requested function is not implemented on this setup
================================================================================
============================ Show Valid mclk Range =============================
GPU[0]		: Requested function is not implemented on this setup
================================================================================
=========================== Show Valid voltage Range ===========================
GPU[0]		: Requested function is not implemented on this setup
================================================================================
============================= Voltage Curve Points =============================
GPU[0]		: Requested function is not implemented on this setup
================================================================================
=============================== Consumed Energy ================================
GPU[0]		: Not supported on the given system
================================================================================
============================= End of ROCm SMI Log ==============================

```
## rocminfo

```
[37mROCk module is loaded[0m
=====================    
HSA System Attributes    
=====================    
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE                              
System Endianness:       LITTLE                             

==========               
HSA Agents               
==========               
*******                  
Agent 1                  
*******                  
  Name:                    AMD Ryzen 5 5625U with Radeon Graphics
  Uuid:                    CPU-XX                             
  Marketing Name:          AMD Ryzen 5 5625U with Radeon Graphics
  Vendor Name:             CPU                                
  Feature:                 None specified                     
  Profile:                 FULL_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        0(0x0)                             
  Queue Min Size:          0(0x0)                             
  Queue Max Size:          0(0x0)                             
  Queue Type:              MULTI                              
  Node:                    0                                  
  Device Type:             CPU                                
  Cache Info:              
    L1:                      32768(0x8000) KB                   
  Chip ID:                 0(0x0)                             
  ASIC Revision:           0(0x0)                             
  Cacheline Size:          64(0x40)                           
  Max Clock Freq. (MHz):   2300                               
  BDFID:                   0                                  
  Internal Node ID:        0                                  
  Compute Unit:            12                                 
  SIMDs per CU:            0                                  
  Shader Engines:          0                                  
  Shader Arrs. per Eng.:   0                                  
  WatchPts on Addr. Ranges:1                                  
  Features:                None
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: FINE GRAINED        
      Size:                    14153264(0xd7f630) KB              
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 2                   
      Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      Size:                    14153264(0xd7f630) KB              
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
    Pool 3                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    14153264(0xd7f630) KB              
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       TRUE                               
  ISA Info:                
*******                  
Agent 2                  
*******                  
  Name:                    gfx90c                             
  Uuid:                    GPU-XX                             
  Marketing Name:          AMD Radeon Graphics                
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        128(0x80)                          
  Queue Min Size:          64(0x40)                           
  Queue Max Size:          131072(0x20000)                    
  Queue Type:              MULTI                              
  Node:                    1                                  
  Device Type:             GPU                                
  Cache Info:              
    L1:                      16(0x10) KB                        
    L2:                      1024(0x400) KB                     
  Chip ID:                 5607(0x15e7)                       
  ASIC Revision:           0(0x0)                             
  Cacheline Size:          64(0x40)                           
  Max Clock Freq. (MHz):   1800                               
  BDFID:                   1024                               
  Internal Node ID:        1                                  
  Compute Unit:            7                                  
  SIMDs per CU:            4                                  
  Shader Engines:          1                                  
  Shader Arrs. per Eng.:   1                                  
  WatchPts on Addr. Ranges:4                                  
  Features:                KERNEL_DISPATCH 
  Fast F16 Operation:      TRUE                               
  Wavefront Size:          64(0x40)                           
  Workgroup Max Size:      1024(0x400)                        
  Workgroup Max Size per Dimension:
    x                        1024(0x400)                        
    y                        1024(0x400)                        
    z                        1024(0x400)                        
  Max Waves Per CU:        40(0x28)                           
  Max Work-item Per CU:    2560(0xa00)                        
  Grid Max Size:           4294967295(0xffffffff)             
  Grid Max Size per Dimension:
    x                        4294967295(0xffffffff)             
    y                        4294967295(0xffffffff)             
    z                        4294967295(0xffffffff)             
  Max fbarriers/Workgrp:   32                                 
  Pool Info:               
    Pool 1                   
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
      Size:                    2097152(0x200000) KB               
      Allocatable:             TRUE                               
      Alloc Granule:           4KB                                
      Alloc Alignment:         4KB                                
      Accessible by all:       FALSE                              
    Pool 2                   
      Segment:                 GROUP                              
      Size:                    64(0x40) KB                        
      Allocatable:             FALSE                              
      Alloc Granule:           0KB                                
      Alloc Alignment:         0KB                                
      Accessible by all:       FALSE                              
  ISA Info:                
    ISA 1                    
      Name:                    amdgcn-amd-amdhsa--gfx90c:xnack-   
      Machine Models:          HSA_MACHINE_MODEL_LARGE            
      Profiles:                HSA_PROFILE_BASE                   
      Default Rounding Mode:   NEAR                               
      Default Rounding Mode:   NEAR                               
      Fast f16:                TRUE                               
      Workgroup Max Size:      1024(0x400)                        
      Workgroup Max Size per Dimension:
        x                        1024(0x400)                        
        y                        1024(0x400)                        
        z                        1024(0x400)                        
      Grid Max Size:           4294967295(0xffffffff)             
      Grid Max Size per Dimension:
        x                        4294967295(0xffffffff)             
        y                        4294967295(0xffffffff)             
        z                        4294967295(0xffffffff)             
      FBarrier Max Size:       32                                 
*** Done ***             


```
## radeontop
```
                                           radeontop 1.4, running on RENOIR bus 04, 120 samples/sec                                            
                                                                        │
                                                  Graphics pipe   0.83% │ 
────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────
                                                   Event Engine   0.00% │
                                                                        │
                                    Vertex Grouper + Tesselator   0.00% │
                                                                        │
                                              Texture Addresser   0.00% │
                                                                        │
                                                  Shader Export   0.00% │
                                    Sequencer Instruction Cache   0.00% │
                                            Shader Interpolator   0.00% │
                                                                        │
                                                 Scan Converter   0.00% │
                                             Primitive Assembly   0.00% │
                                                                        │
                                                    Depth Block   0.00% │
                                                    Color Block   0.00% │
                                                                        │
                                              257M / 2015M VRAM  12.74% │         
                                                45M / 6901M GTT   0.66% │
                                     1.13G / 1.33G Memory Clock  84.52% │                                                          
                                     0.46G / 1.80G Shader Clock  25.74% │                  
                                                                        │

```

# Resources for ROCm on APU

## Bruhnspace

- https://bruhnspace.com/en/bruhnspace-rocm-for-amd-apus/

- Bruhnspace AB, Unibap AB, and Mälardalen University are collaborating on a project for low-latency data processing on embedded systems.  
- The project focuses on AMD's Accelerated Processing Units (APUs) and the AMD professional compute library ROCm.  
- AMD ROCm is a foundation for advanced computing but does not officially support APUs.  
- Bruhnspace provides experimental ROCm packages with APU support for research purposes.  
- They have built and verified ROCm 3.10.0 and 3.3.0 packages that can run on APUs and are available for Ubuntu 20.04 LTS.  
- The packages are built using Docker and should work with any standard 20.04 installation.  
- Bruhnspace provides experimental builds of TensorFlow and pyTorch for AMD APUs.  
- PlaidML works with ROCm for APUs as it has OpenCL support for APUs enabled in the AMD ROCm packages.  
- The ROCm APU packages are strictly for development and testing; there is no warranty or guaranteed support.  
- The project requires strict adherence to Linux kernel versions for APUs: Linux Kernel 4.18 or newer for AMD Carrizo/Bristol Ridge, Kernel 4.19 or newer for AMD Raven Ridge, and Kernel 5.4 or newer for AMD upcoming Renoir.  
- Credits are given to AMD corporation for supporting their project but less so for not delivering promised APU support.

## AMD Ryzen APU turned into a 16GB VRAM GPU and it can run Stable Diffusion

### Reddit post

- The AMD 4600G, priced at $95, includes a 6-core CPU and a 7-core GPU.  
- The 5600G, priced around $130, offers a better CPU but the same GPU as the 4600G.  
- The 4600G can be turned into a 16GB VRAM GPU under Linux.  
- The 4600G functions similarly to discrete AMD GPUs like the 5700XT and 6700XT.  
- It supports the AMD software stack ROCm, and thus also supports AI applications like Pytorch and Tensorflow.  
- The 16GB VRAM capability is significant as it surpasses most discrete GPUs.  
- Despite being slower, the 4600G is beneficial as it prevents out-of-memory errors that can occur if an application requires 12GB or more of VRAM.  
- For stable diffusion, it can generate a 50-step 512x512 image in about 1 minute and 50 seconds, outperforming some high-end CPUs.  
- The 5600G has proven to be a popular product, and users are encouraged to test it.  
- Tutorials for using the 5600G are available on the YouTube channel 'tech-practice9805'.  
- Updates and future content can be found by following the Twitter handle @TechPractice1.

- Youtube channel of this Reddit user: https://www.youtube.com/@tech-practice9805

### Youtube video Democratize AI: turn $95 AMD APU into a 16GB VRAM GPU AI workhorse. Demo of AI app. stable diffusion
- https://www.youtube.com/watch?v=HPO7fu7Vyw4

- Artificial intelligence is transforming industries but access is limited by high costs and technical expertise.  
- GPU VRAM is an important factor for AI applications but consumer GPU prices have skyrocketed.  
- Nvidia has reached a 1 trillion dollar market cap and becomes the world's largest chip company.  
- To democratize AI, low cost consumer hardware and smart designed software are needed.  
- AMD APU is introduced as a budget-friendly hardware solution, combining CPU and GPU on a single die.  
- The least expensive AMD APU is 4600 G priced at 95 US dollars, while the slightly more expensive is 5600g at 127 US Dollars.  
- The GPU can be turned into a 16 gigabytes VRAM GPU.  
- System costs can be as low as 400 US dollars for all brand new components.  
- The power consumption is around 66 Watts for idle system and around 96 Watts at full usage.  
- The operation cost is about 2.4 US dollars for a month or 30 US dollars per year.  
- The system can run mainstream machine learning frameworks such as PyTorch and TensorFlow, and state-of-the-art AI applications.  
- The system has future expandability for more powerful discrete GPU.  
- A demo of running stable diffusion using command line and automatic web UI running stable diffusion is provided.

### https://news.ycombinator.com/item?id=37162762

- AMD Ryzen APU can be converted into a 16GB VRAM GPU.  
- It can run Stable Diffusion, an AI application, which is incredibly VRAM intensive.  
- The APU runs slower, but it means the entire model can be loaded in memory instead of offloading chunks of it.  
- Using DDR5 as VRAM results in slower read/write speeds compared to a proper GPU.  
- Stable Diffusion is VRAM-bandwidth heavy and benefits from the higher-speed GDDR6 or HBM RAM on a high-end GPU.  
- The APU can be useful for running older generation Nvidia gear with 24GB RAM.  
- The APU, while having a large amount of memory, is slower and can be 50x-100x slower than a gamer card from the past couple of years.  
- The potential for AMD to improve their speed with this configuration on later APUs could pose a threat to Nvidia.  
- The 4600G supports two channels of DDR4-3200 which has a maximum memory bandwidth of around 50GB/s, whereas actual graphics cards are in the hundreds.  
- AMD Phoenix APU, specifically 7840H, 7840HS, 7940H, or 7940HS, could be interesting to try as it has up to 54W TDP and supports two channels of DDR5-5600, providing a memory bandwidth of 138 GB/second.  
- The APU can be set to use a certain amount of the system's RAM as VRAM. For example, a system with 32GB of RAM can potentially allocate up to 16GB as VRAM for the APU.  
- The 4600G APU can theoretically achieve 710 FP32 GFlops on CPU and 1702 GFlops on the integrated GPU.  
- AMD now supports APU, making it more accessible for AI applications.  
- The APU can be used to set up a high-VRAM cluster due to its parallelization capabilities.  
- The APU runs Pytorch and Tensorflow, making it useful for running most AI applications.  
- The APU is slower but more affordable, which makes it appealing for those who don't have high-end GPUs.
