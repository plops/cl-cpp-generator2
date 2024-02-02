# Ryzen Monitor README  
   
## Introduction  


Ryzen Monitor (https://github.com/hattedsquirrel/ryzen_monitor) is a
console application that provides real-time insights into power
consumption, voltage, and temperature across the cores of Ryzen
CPUs. The application presents the data in a tabular format, as shown
below:
   
```  
╭─────────┬────────────┬──────────┬─────────┬──────────┬─────────────┬─────────────┬─────────────╮  
│  Core 0 │   Sleeping │  0.008 W │ 0.203 V │  49.95 C │ C0:   0.5 % │ C1:   0.3 % │ C6:  99.2 % │  
│  Core 1 │   Sleeping │  0.074 W │ 0.373 V │  50.17 C │ C0:   3.4 % │ C1:  43.9 % │ C6:  52.7 % │  
...  
╰─────────┴────────────┴──────────┴─────────┴──────────┴─────────────┴─────────────┴─────────────╯  
```  
   
While the application only provides instantaneous readings, the desire
to visualize trends and interactions between CPU cores motivated the
development of this project, which utilizes Dear ImGui to plot these
metrics in real-time graphs.
   
## Features  
- Real-time graphical visualization of power, voltage, and temperature data for Ryzen CPUs.  
- High refresh rate capable of revealing transient spikes during short bursts of computational load.  
- A screenshot of the software's interface is available at `source01/doc/screenshot.png`.  
   
## GitHub Actions  
The `.github/workflows/cmake_143.yml` file sets up a GitHub Action to
compile the binary on Ubuntu 22 with Clang 18. To access the final
build artifact, navigate to: [GitHub Actions
Workflow](https://github.com/plops/cl-cpp-generator2/actions/workflows/cmake_143.yml),
select a successful build, and download the artifact.
   
## Dependencies  
The project relies on several external repositories, which are cloned as follows:
```  
cd ~/src  
git clone https://github.com/glfw/glfw  
git clone https://github.com/orcornut/imgui  
git clone https://github.com/epezent/implot  
git clone https://github.com/hattedsquirrel/ryzen_monitor  
```  
   
Our CMake configuration includes the necessary source files from these
dependencies to produce a statically linked binary. Currently, the
build is configured for GLFW with X11 support, as Wayland support has
not been tested.
   
Regarding ImPlot, the default settings for coordinate precision have
not been modified, but improvements in this area could enhance graph
fidelity.
   
## Future Improvements  
- Optimize data reading code for minimal system impact in terms of
  load, power consumption, and temperature.
- Implement data transmission using gRPC to allow external Python
  tools to perform data analysis and facilitate the benchmarking of
  CPU components.
   
### Placing Load on Different Cores  
Experiment with allocating stress to specific cores using the following command:
```  
for i in `seq 0 2 12`; do taskset -c $i stress -c 1 --timeout 1; done  
```  
   
Subsequent development might focus on a new project designed
exclusively for data collection, eliminating the GUI and OpenGL
dependencies to meet the stated improvement goals.
   
## References  
The development of this code was informed by a review of various
sources, particularly concerning eBPF technology, which offers
intriguing possibilities for correlating CPU measurements with system
calls.

# References:

the following sections contain summaries of websites that i read while developing this code

## Tracing System calls with eBPF

Combining CPU temperature and power metrics with system call data
could reveal how certain operations affect CPU performance.

- https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/

- Introduction to process behavior anomaly detection with eBPF and unsupervised learning using autoencoders.  
- Traditional anomaly detection methods vs. a novel approach that avoids predefined system call lists and considers the rate of system call usage.  
- Explanation of eBPF (Extended Berkeley Packet Filter) technology for sandboxed kernel runtime interception.  
- Description of how eBPF enables innovation within the Linux kernel without modifying source code or loading kernel modules.  
- Usage of the Python BCC (BPF Compiler Collection) package for eBPF program compilation and execution.  
- Details on system call tracing with eBPF using tracepoints or kprobes/kretprobes and challenges associated with argument fetching and buffer throughput.  
- Alternative approach: Histogram-based tracing of `sys_enter` events to circumvent argument reading issues and buffer throughput limitations.  
- Code snippets demonstrating eBPF program setup and user space vector polling to compute the rate of change for every system call.  
- Introduction to autoencoders for anomaly detection without labeled data, capable of internal data representation and output.  
- Neural network architecture for the autoencoder with 512 inputs and outputs and an internal representation layer half the size.  
- Model training using CSV dataset split into training and testing/validation data, with mean square error loss function for reconstruction error.  
- Establishment of a reference error threshold based on the maximum reconstruction error on normal data.  
- Example demonstration using Spotify on Linux as a test case for live process monitoring and anomaly detection during unusual actions like "Connect with Facebook."  
- Instructions for capturing live data, training the model, saving it to a file, and then using it for real-time anomaly detection.  
- Results demonstrating anomaly detection in practice, with output indicating the cumulative error and the most anomalous system calls.  
- Conclusion highlighting the effectiveness of the method, potential for performance improvement, and accuracy tuning.


## https://hattedsquirrel.net/2020/12/power-consumption-of-ryzen-5000-series-cpus/
- Little known about Ryzen 5000 series power consumption in idle and light load scenarios.  
- Analysis conducted using Ryzen 9 5900X.  
- Power efficiency claims by AMD are true under full load, but not in idle or light load.  
- Idle and light load power consumption higher than expected, not reflected in diagnostic software.  
- Ryzen 9 5900X with dedicated GPU power consumption:  
  - 27W in complete idle (monitor off).  
  - ~53W during rapid mouse movement.  
  - 57W when playing a YouTube video.  
- In comparison, Intel i7-4790T system consumed significantly less power for similar tasks.  
- Discrepancies observed between actual power draw and software-reported figures.  
- Power measurements conducted separately on the 8-pin (P8) and 24-pin (P24) motherboard connectors.  
- Idle power draw discrepancy found; higher than reported by software tools.  
- Reverse-engineered the power distribution on a Gigabyte B550M AORUS PRO-P motherboard.  
- Observed high power draw from VDDIO_MEM_S3, which supplies power to the CPU's memory controller.  
- No software accessible measurement for VDDIO_MEM_S3 power draw.  
- Significant power draw not reported in P_Core and P_SoC, suggesting internal CPU components consume more power than reported.  
- "Package Power" and "PPT" values underestimate actual power consumption and thermal output.  
- CPU not ideal for low-power desktop PCs due to high power consumption even under light load.  
- No immediate solutions to reduce power consumption without additional power-saving options.  
- Differences between mobile and desktop SoCs for power management are unclear.

## https://hattedsquirrel.net/2020/12/power-consumption-of-ryzen-5000-series-cpus-part-2/
- Ryzen CPUs consume more power than reported by software.  
- Additional power planes not represented in software but exist in hardware.  
- Reverse engineering of CPU-SMU communication revealed more power counters.  
- Created a CPU performance monitor tool for Ryzen 5000 processors.  
- Actual hardware power measurements differ from software-reported values.  
- For Ryzen 9 5850X: P_Core_sum closer to actual measurement than P_Core.  
- Software-reported power underestimates actual VRM losses and efficiency.  
- Actual power consumption and thermal output much higher than reported package power.  
- Discrepancy between reported and actual power draw raises efficiency questions.  
- Under low load, Ryzen 5000 CPUs are less efficient compared to Intel systems.  
- Linux users have access to more detailed power consumption information.  
- Availability of custom software tool for monitoring power draw.  
- Ryzen 5000 test system uses a Ryzen 9 5850X CPU.  
- Blog post discusses CPU monitor, power draw, and performance under Linux.


## https://hattedsquirrel.net/2021/02/ryzen-cpu-performance-monitor-tool-for-linux/


- Ryzen CPU Performance Monitor Tool for Linux  
  - Updated version of the tool now on GitHub: https://github.com/hattedsquirrel/ryzen_monitor  
  - Supports PM table versions for compatibility with future BIOS updates  
  - Supports multiple Ryzen CPU series, including 3000 and 5000 series  
   
- Installation Steps  
  - Install `ryzen_smu` kernel module from https://gitlab.com/leogx9r/ryzen_smu  
  - Clone and build the `ryzen_monitor` tool:  
    - `git clone https://github.com/hattedsquirrel/ryzen_monitor.git`  
    - `cd ryzen_monitor/`  
    - `make`  
  - Run the monitor tool: `sudo ./src/ryzen_monitor`  
  - Enjoy monitoring your CPU's performance  
   
- Screenshot Provided  
  - Tool showcasing the Ryzen 5000 series CPU performance  
   
- Blog Post Tags  
  - cpu monitor, idle power, linux, performance monitor, power draw, ryzen 5000, ryzen 5900X, ryzen 5950x, zen 3, zen3  
   
- User Feedback (Jens Glathe)  
  - Finds the tool accurate and helpful for explaining thermal instability  
  - Compares favorably with hwinfo64  
  - Provides data on peak power consumption for R9-5950X and R9-3900X  
  - Suggests 10W per core and 40W for the rest as a rough power consumption estimate  
  - Notes peak temperatures can be 5-10 degrees higher than THM  
  - Warns of potential crashes due to throttling issues with inadequate cooling solutions  
  - Thanks the developer for the great work
