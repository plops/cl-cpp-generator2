- i want to have graphs of the data that ryzen monitor shows

# dependencies

```
cd ~/src

git clone https://github.com/glfw/glfw
git clone https://github.com/orcornut/
git clone https://github.com/epezent/implot
```

also see the yaml file for the github action

# Place load on different cores
```
 for i in `seq 0 2 12` ; do taskset -c $i stress -c 1 --timeout 1;done
```

# references:

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
