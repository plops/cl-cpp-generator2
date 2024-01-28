The following summary was created with gpt4-1106 from https://gpuopen.com/wp-content/uploads/2019/11/Vega_7nm_Shader_ISA_26November2019.pdf


Abstract of original document:
   
This document provides a comprehensive guide to the AMD Graphics Core
Next (GCN) processor, a parallel micro-architecture optimized for
computer graphics and general data parallel tasks. The GCN
architecture is adept at handling computationally and
bandwidth-intensive applications due to its array of data-parallel
processors, command processor, memory controller, and other logic
components. The command processor efficiently manages interrupts,
while the memory controller orchestrates direct-memory access and
computes memory-address offsets.
   
The terminology section clarifies key concepts such as wavefronts,
work-items, and the roles of scalar and vector ALUs. The subsequent
chapters delve into the details of program organization, including the
execution of GCN kernels, compute shaders, data sharing, device
memory, and kernel state management. The document thoroughly explains
the scalar and vector ALU operations, scalar and vector memory
operations, flat memory instructions, and data share operations. It
also covers the exporting of pixel and vertex data, providing a
foundation for understanding how the GCN processor handles data flow
and manipulation.
   
Additionally, the document outlines the microcode formats used in the
GCN architecture, ensuring a clear understanding of the encoding and
execution of instructions. It lists all GCN Vega Generation
instructions by format, along with their respective microcode fields
and operational details. This guide serves as an invaluable resource
for developers working with AMD's GCN processor, offering insights
into its capabilities, instruction sets, and programming model for
efficient application development.

- **Chapter 1. Introduction**  
  - AMD GCN processor designed for parallel micro-architecture applications in both computer graphics and general data parallel tasks.  
  - Supports computationally and bandwidth-intensive applications.  
  - Consists of a data-parallel processor array, command processor, memory controller, and other logic.  
  - The command processor manages hardware-generated interrupts to signal command completion.  
  - Memory controller handles direct-memory access, computing memory-address offsets.  
  - Applications on GCN include host processor program and GCN processor kernels.  
  - GCN kernels are managed by host through setting configuration, specifying data domain, cache management, and program execution.  
   
- **1.1. Terminology**  
  - GCN Processor: Scalar and vector ALU for wavefront operations.  
  - Dispatch: Launches a grid of work to the GCN processor.  
  - Workgroup: Collection of wavefronts that synchronize and share data.  
  - Wavefront: 64 work-items executing in parallel on GCN.  
  - Work-item: Individual element of work in the dispatch grid.  
  - Scalar ALU (SALU): Handles one value per wavefront, manages control flow.  
  - Vector ALU (VALU): Executes arithmetic operations for each work-item.  
  - Microcode format: Bit patterns for encoding instructions.  
  - Quad: 2x2 pixel group, relevant for texture mapping.  
  - Texture Sampler and Resource: Describes how textures are read and sampled.  
   
- **Chapter 2. Program Organization**  
  - GCN kernels operate conceptually on every work-item, but execute in groups of 64 (wavefronts).  
  - GCN processor includes scalar and vector ALUs, local data storage, and memory transfer capabilities.  
  - Scalar ALU handles all kernel control flow; vector ALU and memory instructions operate under EXEC mask control.  
  - Vector memory instructions transfer data between VGPRs and memory, unique to each work-item.  
   
- **2.1. Compute Shaders**  
  - Compute kernels are generic programs for data processing on GCN.  
  - Dispatch initiates compute kernels on a grid of work-items.  
  - Work-items have unique addresses within the grid, determine data work and results.  
   
- **2.2. Data Sharing**  
  - GCN processors designed for efficient data sharing between work-items.  
  - Local Data Share (LDS): 64 kB memory space for low-latency communication within work-groups or wavefronts.  
  - Global Data Share (GDS): 64 kB memory used by wavefronts across all compute units for shared control data and complex data structures.  
   
- **2.3. Device Memory**  
  - Offers multiple methods for processing elements to access off-chip memory.  
  - Utilizes L2 read-only cache, L1 cache per compute unit, specific load instructions, and read/write cache with atomic units.  
  - Supports ordered scatter writes and a relaxed consistency model for parallel work-items.

- **Chapter 3. Kernel State**  
  - Describes the kernel states accessible to the shader program.  
   
- **3.1. State Overview**  
  - Lists hardware states that are readable/writable by shader programs.  
  - Includes Program Counter, general-purpose registers (VGPR and SGPR), Local Data Share (LDS), Execute Mask (EXEC), and various status and mode bits.  
   
- **3.2. Program Counter (PC)**  
  - Byte address pointing to the next instruction.  
  - Managed by specific instructions (S_GET_PC, S_SET_PC, S_SWAP_PC) without direct read/write access by the shader program.  
   
- **3.3. EXECute Mask**  
  - 64-bit mask determining active threads for vector instruction execution.  
  - EXECZ is a helper bit indicating if EXEC is zero, used to optimize code skipping.  
   
- **3.4. Status registers**  
  - Read-only fields initialized at wavefront creation, including various control and status flags like SCC, PRIV, TRAP_EN, EXPORT_RDY.  
   
- **3.5. Mode register**  
  - Writable by shader, controls round modes, denormal modes, DX10 clamping, exception enabling, and other operational modes.  
   
- **3.6. GPRs and LDS**  
  - Allocation and access details for Scalar and Vector General-Purpose Registers (SGPRs and VGPRs) and Local Data Share.  
  - Out-of-range behaviors and alignment requirements specified.  
   
- **3.7. M# Memory Descriptor**  
  - Describes the single 32-bit M0 register's multiple uses including addressing, GDS configuration, and message sending.  
   
- **3.8. SCC: Scalar Condition code**  
  - Set by most scalar ALU instructions to indicate results of operations for branching and extended-precision arithmetic.  
   
- **3.9. Vector Compares: VCC and VCCZ**  
  - VCC holds the result of vector ALU comparisons, while VCCZ indicates if VCC is zero.  
   
- **3.10. Trap and Exception registers**  
  - Control and report kernel exceptions, including trap handling and status indication.  
  - Trap temporary SGPRs are privileged and write-accessible only in trap handler mode.  
   
- **3.11. Memory Violations**  
  - Reports memory access errors such as alignment issues and invalid addresses, with sticky bits in TRAPSTS.  
   
- **Chapter 4. Program Flow Control**  
  - Scalar ALU instructions manage program flow including loops, branches, subroutine calls, and traps.  
   
- **4.1. Program Control**  
  - Lists control instructions like S_ENDPGM, S_TRAP, S_SETPRIO, and messaging for host communication.  
   
- **4.2. Branching**  
  - Scalar ALU instructions handle all branching, with conditional branches based on various status bits.  
   
- **4.3. Workgroups**  
  - Work-groups allow synchronization and data sharing among multiple wavefronts on the same compute unit.  
   
- **4.4. Data Dependency Resolution**  
  - Shader hardware mostly resolves data dependencies, but manual S_WAITCNT instructions are needed in certain cases.  
   
- **4.5. Manually Inserted Wait States (NOPs)**  
  - Some dependencies require NOPs or independent instructions to resolve; table provides details for these cases.  
   
- **4.6. Arbitrary Divergent Control Flow**  
  - Handles complex control flow using S_CBRANCH_FORK and S_CBRANCH_JOIN, with stack depth limited to 6 due to thread activity.


- **Chapter 5. Scalar ALU Operations**  
  - SALU operates on one value per wavefront with instructions for integer arithmetic and bitwise operations.  
  - SALU can manipulate the Program Counter and sets the Scalar Condition Code (SCC) for comparison and arithmetic operations.  
   
- **5.1. SALU Instruction Formats**  
  - Five microcode formats for encoding SALU instructions with fields: OP, SDST, SSRC0, SSRC1, SIMM16.  
   
- **5.2. Scalar ALU Operands**  
  - Operands include SGPRs, mode/status/M0 registers, EXEC/VCC masks, SCC, PC, inline constants, hardware registers, and literal constants.  
   
- **5.3. Scalar Condition Code (SCC)**  
  - SCC set by SALU instructions: 1 for true comparisons, arithmetic carry out, non-zero results.  
   
- **5.4. Integer Arithmetic Instructions**  
  - Instructions for addition, subtraction, absolute difference, min/max, multiplication, and sign extension with effects on SCC.  
   
- **5.5. Conditional Instructions**  
  - Use SCC for conditional operations, like selecting between two values or moving data.  
   
- **5.6. Comparison Instructions**  
  - Compare two values and set SCC based on the comparison result.  
   
- **5.7. Bit-Wise Instructions**  
  - Perform bitwise operations without type interpretation; some set SCC if the result is non-zero.  
   
- **5.8. Access Instructions**  
  - Access and modify hardware internal registers.  
   
- **Chapter 6. Vector ALU Operations**  
  - VALU operations perform on 64 threads and write results back to VGPRs, SGPRs, or the EXEC mask.  
   
- **6.1. Microcode Encodings**  
  - Instructions encoded in VOP2, VOP1, VOPC, VINTRP, VOP3, and VOP3P formats.  
   
- **6.2. Operands**  
  - VALU instructions take inputs from VGPRs, SGPRs, inline constants, literal constants, LDS direct data reads, M0, and EXEC mask.  
   
- **6.3. Instructions**  
  - Comprehensive list of VALU instructions by microcode encoding.  
   
- **6.4. Denormalized and Rounding Modes**  
  - Shader program controls rounding and denormalized number handling via MODE register.  
   
- **6.5. ALU Clamp Bit Usage**  
  - "Clamp" bit behavior for V_CMP instructions signals floating point exceptions; clamps integer and floating point operation results.  
   
- **6.6. VGPR Indexing**  
  - Allows M0 register to act as an index into VGPRs for VALU instructions.  
   
- **6.7. Packed Math**  
  - Packed math operations work on two 16-bit values within a Dword using VOP3P encoding.  
   
- **Chapter 7. Scalar Memory Operations**  
  - SMEM instructions allow loading data from memory into SGPRs or writing SGPR data to memory.  
   
- **7.1. Microcode Encoding**  
  - SMEM instruction encoding with fields for opcode, immediate offsets, globally coherent flag, data, base address, and scalar offset enable.  
   
- **7.2. Operations**  
  - Load and store instructions for scalar memory reads/writes; atomic operations; cache invalidation/write-back; clock and real-time counter reads.  
   
- **7.3. Dependency Checking**  
  - LGKM_CNT counter ensures data has returned to SGPRs before continuing.  
   
- **7.4. Alignment and Bounds Checking**  
  - Restrictions for SDST and SBASE alignment; bounds checking for memory address prevents out-of-range operations.


- **Chapter 8. Vector Memory Operations**  
  - VMEM instructions read/write data per work-item in wavefront to/from VGPRs, contrast to Scalar Memory which shares data across a wavefront.  
  - VMEM operations include MTBUF, MUBUF, and MIMG instructions.  
  - Memory buffer descriptors (V# or T#) define address and characteristics of memory buffers for these operations.  
   
- **8.1. Vector Memory Buffer Instructions**  
  - Transfer data between VGPRs and buffer objects via texture cache.  
  - Buffer reads can return data to VGPRs or LDS, supporting heterogeneous data, but no texel filtering.  
  - Two types of buffer instructions: MUBUF (untyped) and MTBUF (typed).  
  - Atomic operations perform arithmetic on data in memory, optional return of pre-operation value.  
   
- **8.1.1. Simplified Buffer Addressing**  
  - Hardware calculates memory address for buffer access using base address, index, and offset.  
   
- **8.1.2. Buffer Instructions**  
  - Read/write operations for linear buffers in memory, supporting data as small as one byte up to four Dwords.  
  - D16 variants convert results to packed 16-bit values.  
   
- **8.1.3. VGPR Usage**  
  - VGPRs supply address and data for buffer instructions, with zero, one, or two VGPRs used depending on the operation.  
   
- **8.1.4. Buffer Data**  
  - Data type and amount read/written controlled by data-format, numeric-format, opcode, and destination-component-selects.  
   
- **8.1.5. Buffer Addressing**  
  - Addressing combines base address, index, and offset; different for linearly addressed and swizzled buffers.  
   
- **8.1.6. 16-bit Memory Operations**  
  - D16 buffer instructions load/store 16 bits per work item, with different variants for lower and upper 16 bits.  
   
- **8.1.7. Alignment**  
  - Requires Dword alignment for Dword or larger reads/writes.  
   
- **8.1.8. Buffer Resource**  
  - Descriptor defines buffer location, data format, and characteristics; specified in SGPRs.  
   
- **8.1.9. Memory Buffer Load to LDS**  
  - Subset of MUBUF instructions allow direct read from memory buffer into LDS, bypassing VGPRs.  
   
- **8.1.10. GLC Bit Explained**  
  - GLC bit affects cache behavior for loads, stores, and atomics, including whether pre-operation values are returned.  
   
- **8.2. Vector Memory (VM) Image Instructions**  
  - Transfer data between VGPRs and memory through texture cache for homogeneous data in image objects.  
  - Use image resource (T#) and sampler (S#) constants; IMAGE_LOAD, SAMPLE_* instructions fetch and optionally filter data.  
   
- **8.2.1. Image Instructions**  
  - Includes SAMPLE_* and IMAGE_LOAD/STORE/ATOMIC operations with specified microcode fields.  
   
- **8.3. Image Opcodes with No Sampler**  
  - For non-sampled image opcodes, all VGPR address values are treated as unsigned integers.  
   
- **8.4. Image Opcodes with a Sampler**  
  - For sampled image opcodes, all VGPR address values are treated as floats, with additional values required for certain operations.  
   
- **8.4.1. VGPR Usage**  
  - VGPRs hold address, data, and return data for image operations; data is determined by DMASK field in instruction.  
   
- **8.4.2. Image Resource**  
  - Defines image buffer location, dimensions, tiling, and data format in memory; specified in SGPRs.  
   
- **8.4.3. Image Sampler**  
  - Defines operations on texture map data such as clamping and filtering; specified in SGPRs.  
   
- **8.4.4. Data Formats**  
  - Lists available data formats for image and buffer resources.  
   
- **8.4.5. Vector Memory Instruction Data Dependencies**  
  - Shader developer responsibility to handle data hazards and VMEM instruction completion waits.  
   
- **Chapter 9. Flat Memory Instructions**  
  - Flat Memory instructions read/write data for each work-item using a single flat address without resource constants.  
   
- **9.1. Flat Memory Instruction**  
  - Instructions include Flat, Scratch, and Global operations with specific microcode formats and opcodes.  
   
- **9.2. Instructions**  
  - Similar to Buffer instruction set, FLAT instructions require an SGPR-pair for scratch-space information.  
   
- **9.3. Addressing**  
  - Supports both 64- and 32-bit addressing with specific computation for scratch space.  
   
- **9.4. Global**  
  - Similar to Flat, but ensures no threads access LDS space, offering two types of addressing.  
   
- **9.5. Scratch**  
  - Similar to Flat, ensures no LDS space access, supports multi-Dword and misaligned access.  
   
- **9.6. Memory Error Checking**  
  - Detects errors due to invalid address, write to read-only surface, misalignment, or out-of-range address.  
   
- **9.7. Data**  
  - Specifies data handling in VGPRs, with no data-format conversion for FLAT instructions.  
   
- **9.8. Scratch Space (Private)**  
  - Defines area of memory for thread-private data with hardware-computed additional address information.  
   
- **Chapter 10. Data Share Operations**  
  - LDS is a low-latency RAM for data sharing between work-items with high effective bandwidth compared to global memory.  
   
- **10.1. Overview**  
  - LDS is an on-chip scratchpad for temporary data located next to ALUs, supporting simultaneous access and helping to avoid bank conflicts.  
   
- **10.2. Dataflow in Memory Hierarchy**  
  - Conceptual diagram of dataflow within GPU memory structure, highlighting the role of LDS.  
   
- **10.3. LDS Access**  
  - Access methods include direct reads, parameter reads, and indexed or atomic operations.  
   
- **10.3.1. LDS Direct
Reads**  
  - VALU instructions can broadcast a single DWORD from LDS to all threads in the wavefront.  
   
- **10.3.2. LDS Parameter Reads**  
  - Pixel shaders use LDS to read vertex parameter values for interpolation.  
   
- **10.3.3. Data Share Indexed and Atomic Access**  
  - LDS and GDS support indexed and atomic operations, providing unique addresses and data per work-item from VGPRs.  
   
- **10.3.4. LDS Instruction Fields**  
  - Describes the fields in LDS instructions including opcode, address, data sources, and destination VGPR.  
   
- **10.3.5. LDS Indexed Load/Store**  
  - Provides load and store operations to read data from, and write data to, LDS with various data sizes.  
   
- **10.3.6. LDS Atomic Ops**  
  - Describes atomic operations on LDS, including swap, add, subtract, min, max, and bitwise operations.  
   
- **10.4. Scratch Space (Private)**  
  - Details on using scratch space for private memory with hardware-computed address adjustments.   
  
- **Chapter 11. Exporting Pixel and Vertex Data**  
  - Export instructions transfer shader data from VGPRs to output buffer.  
  - Data types for export: Vertex Position, Vertex Parameter, Pixel color, Pixel depth (Z).  
  - Export instruction uses EXP microcode format.  
   
- **11.1. Microcode Encoding**  
  - VM bit indicates if EXEC mask represents the valid-mask for a wavefront.  
  - DONE bit marks the final pixel shader or vertex-position export.  
  - COMPR bit indicates if the data is compressed (16-bits per component).  
  - TARGET field designates the type of data exported (MRT, Z, Null, Position, Param).  
  - EN field specifies enabling of export components.  
   
- **11.2. Operations**  
  - **11.2.1. Pixel Shader Exports**  
    - Export instructions copy color data to MRTs, optionally output depth (Z) data.  
    - Pixel shader must use at least one export instruction; the last must have DONE bit set.  
    - EXEC mask applied to all exports; results are accumulated in output buffer.  
  - **11.2.2. Vertex Shader Exports**  
    - Vertex shader exports position and parameter data for subsequent pixel shaders.  
    - Must output at least one position vector, with the last position export having DONE bit set.  
   
- **11.3. Dependency Checking**  
  - Export instructions executed in two phases with the use of EXPCNT and S_WAITCNT.  
  - S_WAITCNT prevents shader from overwriting data before export completion.  
  - Exports of the same type are completed in order, others can be out of order.  
  - SKIP_EXPORT bit in STATUS register treats EXPORT instructions as NOPs.  
   
- **Chapter 12. Instructions**  
  - Lists all GCN Vega Generation instructions by format.  
  - Definitions for instruction suffixes (e.g., B32, F64, I8, U16) provided.  
  - Abbreviations used in definitions (e.g., D = destination, U = unsigned integer) explained.  
  - Note on rounding and Denormal modes for floating-point operations.  
    
- **12.1. SOP2 Instructions**  
  - Example: `S_ADD_U32` adds two unsigned 32-bit integers with flags for overflow/carry-out.  
    
- **12.2. SOPK Instructions**  
  - Example: `S_MOVK_I32` moves a sign-extended 16-bit constant into a destination register.  
    
- **12.3. SOP1 Instructions**  
  - Example: `S_MOV_B32` moves a 32-bit unsigned value into the destination register.  
   
- **12.4. SOPC Instructions**  
  - Example: `S_CMP_EQ_I32` sets the scalar condition code if two 32-bit integers are equal.  
   
- **12.5. SOPP Instructions**  
  - Example: `S_BRANCH` performs a relative branch based on a signed DWORD offset.  
   
- **12.6. SMEM Instructions**  
  - Example: `S_LOAD_DWORD` reads a 32-bit value from the scalar data cache.  
   
- **12.7. VOP2 Instructions**  
  - Example: `V_ADD_F32` adds two single-precision floating-point values.  
   
- **12.8. VOP1 Instructions**  
  - Example: `V_MOV_B32` moves a 32-bit unsigned value into the destination VGPR.  
   
- **12.9. VOPC Instructions**  
  - Example: `V_CMP_EQ_F32` compares two single-precision floating-point values for equality.  
   
- **12.10. VOP3P Instructions**  
  - Example: `V_PK_MAD_I16` performs a packed multiply-add operation on 16-bit integers.  
   
- **12.11. VINTERP Instructions**  
  - Example: `V_INTERP_P1_F32` interpolates the P1 parameter for a pixel shader.  
   
- **12.12. VOP3A & VOP3B Instructions**  
  - Example: `V_MAD_F32` performs a multiply-add operation on single-precision floating-point values.  
   
- **12.13. LDS & GDS Instructions**  
  - Example: `DS_ADD_U32` performs an atomic add operation on a 32-bit unsigned value in LDS/GDS memory.  
   
- **12.14. MUBUF Instructions**  
  - Example: `BUFFER_LOAD_FORMAT_X` loads a single dword from a buffer with format conversion.  
   
- **12.15. MTBUF Instructions**  
  - Example: `TBUFFER_LOAD_FORMAT_X` loads a single dword from a typed buffer with format conversion.  
   
- **12.16. MIMG Instructions**  
  - Example: `IMAGE_LOAD` loads image data with format conversion specified by the resource constant.  
   
- **12.17. EXPORT Instructions**  
  - Example: `EXP` exports pixel or vertex data to the appropriate output buffer.  
   
- **12.18. FLAT, Scratch and Global Instructions**  
  - Example: `FLAT_LOAD_DWORD` loads a dword from flat-addressed memory.  
   
- **12.19. Instruction Limitations**  
  - DPP not supported by certain instructions like `V_MADMK_F32`.  
  - SDWA not supported by certain instructions like `V_MAC_F32`.
  

- **Chapter 13. Microcode Formats**  
  - Microcode formats and instruction formats outlined for GCN architecture.  
  - Little-endian byte and bit-ordering are used for memory and registers.  
  - Summary of microcode formats and widths provided in Table 52.  
  - Scalar ALU and Control Formats include SOP2, SOP1, SOPK, and SOPP with various opcodes.  
  - Scalar Memory Format (SMEM) with 64-bit width.  
  - Vector ALU Formats include VOP1, VOP2, VOPC, VOP3A, VOP3B, VOP3P with different widths.  
  - Vector Memory Buffer Formats include MTBUF and MUBUF with 64-bit width.  
  - Vector Memory Image Format (MIMG) and Export Format (EXP) detailed.  
  - Flat Formats for memory access include FLAT, GLOBAL, SCRATCH.  
  - Definitions use notation such as int(2) for a two-bit unsigned integer field and enum(7) for a seven-bit enumerated set of values.  
  - Instruction suffixes indicate data type and size (e.g., F32 for 32-bit floating-point).  
  - Scalar ALU and Control Formats provide details on SOP2, SOPK, SOP1, SOPC, SOPP instruction formats.  
  - Vector ALU Formats provide details on VOP2, VOP1, VOPC, VOP3A, VOP3B, VOP3P instructions.  
  - SDWA and DPP formats are defined for specific data operations.  
  - Vector Parameter Interpolation Format (VINTRP) outlined.  
  - LDS/GDS Format (DS) for Local and Global Data Sharing instructions detailed.  
  - Vector Memory Buffer Formats (MTBUF and MUBUF) provide typed and untyped buffer access.  
  - Vector Memory Image Format (MIMG) instructions described.  
  - Export Format (EXP) for EXPORT instructions.  
  - Flat Formats (FLAT, GLOBAL, SCRATCH) for different memory segment access outlined.  
