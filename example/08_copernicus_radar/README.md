# Copernicus Sentinel-1 SAR Raw Data Processor

A code generation project for processing Sentinel-1 satellite raw radar data, with capabilities for decoding, visualization, and GPU-accelerated SAR image processing.

## Overview

This project uses Common Lisp code generators (`cl-cpp-generator2` and `cl-py-generator`) to produce C++ and Python code for processing Sentinel-1 Level 0 RAW data files from the European Space Agency's Copernicus program. It handles space packet decoding, BAQ decompression, calibration pulse extraction, and SAR image formation.  

## Project Structure

The project consists of four main generator files with different priorities:

### Core Components

1. **gen00.lisp** (Priority 0) - Space Packet Decoder
   - Decodes space packet data and header information
   - Handles packet primary headers and data fields
   - Extracts calibration and echo data  

2. **gen01py.lisp** (Priority 1) - Visualization
   - Visualizes decoded calibration and echo data with Python
   - Generates plotting scripts for data analysis  

3. **gen02.lisp** (Priority 1) - Doppler Processing
   - Attempts Doppler decoding of raw data
   - Processes radar returns for motion analysis  

4. **gen03py.lisp** (Priority 1) - GPU Acceleration
   - Uses Numba and CuPy for GPU-accelerated convolution
   - Optimizes SAR image processing performance  

## Generated Code

### C++ Source Files (source/)

The generators produce modular C++ code for data processing:  

### Doppler Processing (source_doppler/)

Additional CUDA-accelerated code for real-time visualization:  

## Features

- **Space Packet Parsing**: Decodes CCSDS space packet headers and data fields with bit-level precision
- **BAQ Decompression**: Handles Block Adaptive Quantization compressed radar data
- **Calibration Processing**: Extracts and processes multiple calibration pulse types (TX, RX, EPDN, TA, APDN, TXH ISO)
- **Memory-Mapped I/O**: Efficient handling of large RAW data files
- **GPU Acceleration**: CUDA-based processing for real-time performance
- **Visualization**: Comprehensive plotting of radar data and metadata

## Data Processing

The system processes Sentinel-1 data through several stages:

1. Memory-map the raw .dat file
2. Collect packet headers and offsets
3. Decode space packet metadata
4. Extract calibration pulses (with polarity compensation)
5. Decompress BAQ-encoded echo data
6. Store range-compressed data
7. Apply range/azimuth compression for SAR image formation



## Notes

- The code generators use feature flags for debugging (`:safety`, `:nolog`, `:log-brc`, `:log-consume`)
- Supports multiple Sentinel-1 acquisition modes (IW, EW, SM, WV)
- Processes HH, HV, VH, and VV polarization modes
- Handles dihedral reflections from ships for best SNR detection

## Usage

1. Run the Lisp generators to produce C++ and Python code
2. Compile the C++ modules
3. Execute the main processor with a Sentinel-1 RAW .dat file
4. Use the Python scripts to visualize results  


Based on my analysis of the code, here's a detailed breakdown of the four main generator files in the Copernicus radar processing example:

## **gen00.lisp** - Space Packet Decoder Generator (C++)

**Purpose:** This is the priority 0 generator that creates C++ code to decode Sentinel-1 SAR Level-0 data, parsing space packet protocol data units and extracting calibration and echo information.  

**Key Data Structures:**

1. **Space Packet Definition** - A comprehensive 62-byte header structure with bit-level field definitions including packet metadata, timing information, and radar configuration parameters.  

2. **Sequential Bit Reader** - A structure for reading individual bits from the data stream during BAQ (Block Adaptive Quantization) decompression.  

3. **Sub-commutated Ancillary Data** - A 65-word array structure for slowly-updated satellite position, velocity, quaternion, angular rates, and temperature data.  

**Algorithms:**

1. **Huffman Decoders** - Five different Huffman tree decoders (BRC0-BRC4) for different bit rate codes used in BAQ compression.  

2. **BAQ Decompression** - Multiple reconstruction algorithms supporting BAQ modes 0, 3, 4, and 5 with simple and normal reconstruction laws using threshold indices and normalized reconstruction levels.  

3. **Bit Rate Code Extraction** - Decodes 3-bit BRC values that determine which Huffman decoder to use for each 128-sample block.  

**Processing Steps:**

1. Memory-maps the input file for efficient access
2. Collects all space packet headers by scanning for sync markers and extracting 68-byte headers
3. Separates calibration packets from echo packets using the calibration flag
4. Decodes BAQ-compressed I/Q samples for both even/odd real/imaginary components
5. Applies reconstruction laws based on threshold indices and sigma factors
6. Outputs calibration data and aligned range-compressed echo data  

## **gen01py.lisp** - Python Visualization Generator

**Purpose:** Generates Python code to visualize and analyze decoded calibration and echo data, performing chirp fitting and pulse compression analysis.  

**Key Data Structures:**

1. **Pandas DataFrames** - Three main dataframes (df, dfa, dfc) for range data, all packets, and calibration packets with categorical descriptors for polarization, receive channel, and signal types.  

2. **Decimation Filter Parameters** - Arrays defining bandwidth, L/M ratios, filter lengths, and output offsets for 12 different swath configurations.  

3. **Calibration Pulse Arrays** - Separate arrays for tx_cal, rx_cal, epdn_cal, ta_cal, apdn_cal, and txh_iso_cal pulses extracted from calibration packets.  

**Algorithms:**

1. **Sample Rate Computation** - Calculates decimation filter output rates (fdec) and determines N3_tx/N3_rx sample counts based on chirp parameters.  

2. **Chirp Polynomial Fitting** - Uses Chebyshev polynomials to fit both magnitude and phase of replica chirps with Savitzky-Golay filtering.  

3. **Replica Generation** - Computes complex chirp replicas from tx pulse parameters (txprr, txpsf, txpl) for matched filtering.  

4. **Pulse Compression** - FFT-based convolution with conjugated chirp kernel for range compression.  

**Processing Steps:**

1. Loads CSV files containing packet metadata
2. Enriches dataframes with descriptive labels and computed parameters
3. Retrieves ranked pulse configurations from the past (handling rank parameter)
4. Extracts and processes calibration pulses with differential measurements
5. Fits magnitude and phase polynomials to calibration replicas
6. Generates synthetic chirps and performs range compression on image data  

## **gen02.lisp** - CUDA GPU Processing with GUI Generator (C++)

**Purpose:** Generates C++ code with CUDA support for GPU-accelerated doppler processing, including an interactive OpenGL/ImGui GUI for real-time visualization.  

**Key Data Structures:**

1. **CUDA Complex Type** - Uses float2 (Complex) for GPU-compatible complex arithmetic with custom multiplication operations.  

2. **Pulse Configuration Arrays** - Static arrays containing txprr, txpl, txpsf, fdec parameters for each echo line extracted from CSV data.  

3. **GUI State** - GLFW window with ImGui interface including sliders for selecting range lines and plots for raw/processed data visualization.  

**Algorithms:**

1. **CUDA Pointwise Multiplication Kernel** - GPU kernel for element-wise complex multiplication of FFT spectra.  

2. **Chirp Kernel Generation** - Computes conjugated chirp on CPU based on pulse parameters, then uploads to GPU for convolution.  

3. **cuFFT-based Convolution** - Forward FFT of signal and kernel, pointwise multiplication, inverse FFT for matched filtering.  

**Processing Steps:**

1. Memory-maps the range-compressed data file
2. Initializes CUDA device and cuFFT plans
3. For each selected echo line, transfers data to GPU
4. Generates chirp kernel based on line-specific tx parameters
5. Performs FFT-based pulse compression on GPU
6. Displays results in real-time GUI with magnitude and phase plots
7. Allows interactive exploration through slider controls  

## **gen03py.lisp** - Numba/CuPy GPU Compression Generator (Python)

**Purpose:** Generates Python code using Numba and CuPy for GPU-accelerated doppler centroid estimation and azimuth compression with batch processing.  

**Key Data Structures:**

1. **CuPy Memory-Mapped Arrays** - GPU-accessible views of large SAR datasets for batch processing without full loading into RAM.  

2. **Doppler Frequency Grid** - A 2D grid with range in one dimension and doppler frequency shift in the other for generating chirp banks.  

3. **Conditional CSV Loading** - Checks for pre-computed dataframes and only recalculates if missing, saving processing time.  

**Algorithms:**

1. **Doppler Chirp Bank** - Generates a 2D array of chirps with varying doppler frequency offsets for azimuth compression.  

2. **Batch FFT Convolution** - Uses CuPy's FFT along specified axes to compress multiple range lines simultaneously.  

3. **Doppler Centroid Estimation** - Computes phase correlation between adjacent azimuth samples and applies Savitzky-Golay filtering for noise reduction.  

**Processing Steps:**

1. Conditionally loads or recalculates packet metadata and pulse parameters
2. Memory-maps calibration and echo data files
3. Transfers data batches to GPU using Numba CUDA
4. Generates chirp replicas with doppler frequency variations
5. Performs 2D FFT-based azimuth compression
6. Estimates doppler centroid from azimuth phase correlation
7. Applies savgol filtering to smooth doppler estimates
8. Visualizes results with matplotlib  

## Notes

All four generators work together in a processing pipeline:
- **gen00** extracts and decodes raw Level-0 data
- **gen01** analyzes calibration data and generates reference chirps  
- **gen02** provides interactive GPU processing with visualization
- **gen03** performs batch GPU processing for doppler analysis

The generators use Common Lisp macros to generate either C++ or Python code, demonstrating a meta-programming approach where the actual radar processing algorithms are expressed at a higher abstraction level. Each generator creates complete, executable programs with proper memory management, error handling, and optimized data structures for their specific processing tasks.



