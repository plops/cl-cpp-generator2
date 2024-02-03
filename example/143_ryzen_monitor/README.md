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
   
  ![Screenshot](https://raw.githubusercontent.com/plops/cl-cpp-generator2/master/example/143_ryzen_monitor/source01/doc/screenshot.png)


The y-axis represents time in seconds. A stress test was conducted on
each of the 12 threads of a Ryzen 5625U CPU. The graph reflects
activity across 6 distinct cores. Although the Ryzen System Management
Unit (SMU) indicates 8 cores, two are disabled and do not perform
computations. However, these inactive cores still record temperature
data.

   
## GitHub Actions  
The `.github/workflows/cmake_143.yml` file sets up a GitHub Action to
compile the binary on Ubuntu 22 with Clang 18. To access the final
build artifact, navigate to: [GitHub Actions
Workflow](https://github.com/plops/cl-cpp-generator2/actions/workflows/cmake_143.yml),
select a successful build, and download the artifact.
   
## Build Dependencies  
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
   
## Dependencies for code coverage

I installed the dependencies for code coverage in Gentoo like this:

```
 sudo emerge -av lcov  dev-util/gcovr
```

## Runtime Dependencies

On my cpu I need this kernel module. Unfortunately, the newest commits
don't work anymore. I have to use this particular commit from Sun Dec
5 17:29:33 2021, because that one works. I haven't looked into why
newer commits don't work.


```
git clone https://gitlab.com/leogx9r/ryzen_smu.git ryzen_smu
cd ryzen_smu
git checkout de976a9b43d629b7fa0c32e5124bc28bc91d47ef  
```

## Future Improvements  
- Optimize data reading code for minimal system impact in terms of
  load, power consumption, and temperature.
- Implement data transmission using gRPC to allow external Python
  tools to perform data analysis and facilitate the benchmarking of
  CPU components.
  
Here is a die shot of a Ryzen 5600G CPU:

![Die Shot of Ryzen CPU](https://raw.githubusercontent.com/plops/cl-cpp-generator2/master/example/143_ryzen_monitor/source01/doc/die_shot_from_techpowerup.jpg) 

It would be interesting to see if the Zen3 core closest to the Vega
Compute units gets heated with GPU load.
   
### Placing Load on Different Cores  
Experiment with allocating stress to specific cores using the following command:
```  
for i in `seq 0 2 12`; do taskset -c $i stress -c 1 --timeout 1; done  
```  
   
Subsequent development might focus on a new project designed
exclusively for data collection, eliminating the GUI and OpenGL
dependencies to meet the stated improvement goals.
   
# References

The development of this code was informed by a review of various
sources.



## https://hattedsquirrel.net/2020/12/power-consumption-of-ryzen-5000-series-cpus/
- Little known about Ryzen 5000 series power consumption in idle and
  light load scenarios.
- Analysis conducted using Ryzen 9 5900X.
- Power efficiency claims by AMD are true under full load, but not in
  idle or light load.
- Idle and light load power consumption higher than expected, not
  reflected in diagnostic software.
- Ryzen 9 5900X with dedicated GPU power consumption:
  - 27W in complete idle (monitor off).
  - ~53W during rapid mouse movement.
  - 57W when playing a YouTube video.
- In comparison, Intel i7-4790T system consumed significantly less
  power for similar tasks.
- Discrepancies observed between actual power draw and
  software-reported figures.
- Power measurements conducted separately on the 8-pin (P8) and 24-pin
  (P24) motherboard connectors.
- Idle power draw discrepancy found; higher than reported by software
  tools.
- Reverse-engineered the power distribution on a Gigabyte B550M AORUS
  PRO-P motherboard.
- Observed high power draw from VDDIO_MEM_S3, which supplies power to
  the CPU's memory controller.
- No software accessible measurement for VDDIO_MEM_S3 power draw.
- Significant power draw not reported in P_Core and P_SoC, suggesting
  internal CPU components consume more power than reported.
- "Package Power" and "PPT" values underestimate actual power
  consumption and thermal output.
- CPU not ideal for low-power desktop PCs due to high power
  consumption even under light load.
- No immediate solutions to reduce power consumption without
  additional power-saving options.
- Differences between mobile and desktop SoCs for power management are
  unclear.

## https://hattedsquirrel.net/2020/12/power-consumption-of-ryzen-5000-series-cpus-part-2/
- Ryzen CPUs consume more power than reported by software.
- Additional power planes not represented in software but exist in
  hardware.
- Reverse engineering of CPU-SMU communication revealed more power
  counters.
- Created a CPU performance monitor tool for Ryzen 5000 processors.
- Actual hardware power measurements differ from software-reported
  values.
- For Ryzen 9 5850X: P_Core_sum closer to actual measurement than
  P_Core.
- Software-reported power underestimates actual VRM losses and
  efficiency.
- Actual power consumption and thermal output much higher than
  reported package power.
- Discrepancy between reported and actual power draw raises efficiency
  questions.
- Under low load, Ryzen 5000 CPUs are less efficient compared to Intel
  systems.
- Linux users have access to more detailed power consumption
  information.
- Availability of custom software tool for monitoring power draw.
- Ryzen 5000 test system uses a Ryzen 9 5850X CPU.
- Blog post discusses CPU monitor, power draw, and performance under
  Linux.


## https://hattedsquirrel.net/2021/02/ryzen-cpu-performance-monitor-tool-for-linux/


- Ryzen CPU Performance Monitor Tool for Linux
  - Updated version of the tool now on GitHub:
    https://github.com/hattedsquirrel/ryzen_monitor
  - Supports PM table versions for compatibility with future BIOS
    updates
  - Supports multiple Ryzen CPU series, including 3000 and 5000 series
 
- Installation Steps
  - Install `ryzen_smu` kernel module from
    https://gitlab.com/leogx9r/ryzen_smu
  - Clone and build the `ryzen_monitor` tool:
    - `git clone https://github.com/hattedsquirrel/ryzen_monitor.git`
    - `cd ryzen_monitor/`
    - `make`
  - Run the monitor tool: `sudo ./src/ryzen_monitor`
  - Enjoy monitoring your CPU's performance
 
- Screenshot Provided
  - Tool showcasing the Ryzen 5000 series CPU performance
 
- Blog Post Tags
  - cpu monitor, idle power, linux, performance monitor, power draw,
    ryzen 5000, ryzen 5900X, ryzen 5950x, zen 3, zen3
 
- User Feedback (Jens Glathe)
  - Finds the tool accurate and helpful for explaining thermal
    instability
  - Compares favorably with hwinfo64
  - Provides data on peak power consumption for R9-5950X and R9-3900X
  - Suggests 10W per core and 40W for the rest as a rough power
    consumption estimate
  - Notes peak temperatures can be 5-10 degrees higher than THM
  - Warns of potential crashes due to throttling issues with
    inadequate cooling solutions
  - Thanks the developer for the great work


## Tracing System calls with eBPF

Combining CPU temperature and power metrics with system call data
could reveal how certain operations affect CPU performance.

- https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/

- Introduction to process behavior anomaly detection with eBPF and
  unsupervised learning using autoencoders.
- Traditional anomaly detection methods vs. a novel approach that
  avoids predefined system call lists and considers the rate of system
  call usage.
- Explanation of eBPF (Extended Berkeley Packet Filter) technology for
  sandboxed kernel runtime interception.
- Description of how eBPF enables innovation within the Linux kernel
  without modifying source code or loading kernel modules.
- Usage of the Python BCC (BPF Compiler Collection) package for eBPF
  program compilation and execution.
- Details on system call tracing with eBPF using tracepoints or
  kprobes/kretprobes and challenges associated with argument fetching
  and buffer throughput.
- Alternative approach: Histogram-based tracing of `sys_enter` events
  to circumvent argument reading issues and buffer throughput
  limitations.
- Code snippets demonstrating eBPF program setup and user space vector
  polling to compute the rate of change for every system call.
- Introduction to autoencoders for anomaly detection without labeled
  data, capable of internal data representation and output.
- Neural network architecture for the autoencoder with 512 inputs and
  outputs and an internal representation layer half the size.
- Model training using CSV dataset split into training and
  testing/validation data, with mean square error loss function for
  reconstruction error.
- Establishment of a reference error threshold based on the maximum
  reconstruction error on normal data.
- Example demonstration using Spotify on Linux as a test case for live
  process monitoring and anomaly detection during unusual actions like
  "Connect with Facebook."
- Instructions for capturing live data, training the model, saving it
  to a file, and then using it for real-time anomaly detection.
- Results demonstrating anomaly detection in practice, with output
  indicating the cumulative error and the most anomalous system calls.
- Conclusion highlighting the effectiveness of the method, potential
  for performance improvement, and accuracy tuning.


## Unit tests and code coverage

- https://dev.to/askrodney/cmake-coverage-example-with-github-actions-and-codecovio-5bjp

- Rodney Labexplains how to set up code coverage for a C++ project
  using CMake with GitHub Actions and codecov.io.
- The project used as an example is an arkanoid-clone game, with
  Catch2 unit tests integrated.
- Catch2 tests are added to the CMake project and configured to run
  automatically.
- A custom target for code coverage is added to the CMake
  configuration, using tools like gcov, lcov, and genhtml.
- A GitHub Action is created to build the project, run tests, and
  upload coverage reports to codecov.io on every push or pull request.
- The GitHub Action includes steps for installing dependencies,
  configuring the build environment, running tests, generating
  coverage reports, and uploading to codecov.io.
- A `requirements.txt` file is needed for installing Python packages
  like gcovr, which are used to convert coverage data to XML.
- Code coverage results are visible on codecov.io and within GitHub
  pull requests.
- Rodney Lab invites feedback, suggestions for improvements, and
  sharing of the article on social media.
- The post includes a call for contributions and interaction with the
  community on various topics related to game development.


### About gcov, lcov and genhtml

The purpose and relationship of gcov, lcov, and genhtml are as
follows:
 
**gcov:** gcov is a test coverage analysis tool provided by the GNU
Compiler Collection (GCC). It helps programmers analyze their code to
identify portions that have not been tested, which can guide the
creation of additional tests to improve code quality. gcov gathers
data on how often each line of code executes during a program's
run. This information is useful for optimizing code and ensuring that
it behaves as expected. For meaningful results, code should be
compiled with specific flags (e.g., `-fprofile-arcs -ftest-coverage`)
and without optimization.
 
**lcov:** lcov is a graphical front-end for gcov. It collects gcov
data for multiple source files and creates a more human-readable
report in HTML format. lcov provides a clearer overview of code
coverage by presenting the data with visual aids such as colored lines
indicating the level of coverage for each line of code. lcov can
filter out irrelevant files from the coverage report, combine coverage
data from different test runs, and remove coverage data that is not of
interest.
 
**genhtml:** genhtml is a tool that is part of the lcov suite. It
takes the coverage data files produced by lcov and generates an HTML
view of the coverage. This HTML report includes detailed information
about which lines of code are covered by tests and which are not, and
it uses color-coding to indicate the level of coverage. genhtml
provides various options for customizing the output, including
changing the color thresholds, adding a title or description, and
adjusting the layout through CSS.
 
**Relationship:** The relationship between these tools is that gcov is
used to collect the raw coverage data at the lowest level. lcov then
processes this data, allowing for easier analysis and management of
coverage information. Finally, genhtml takes the processed data from
lcov and creates a visual HTML report that developers can use to
improve their code's quality and test coverage. The workflow typically
involves running tests with gcov instrumentation, using lcov to gather
and refine the data, and then using genhtml to generate an accessible
report for developers to review.


### About gcovr

`gcovr` is a command-line Python tool that provides a readable summary
of code coverage statistics for a program, helping you identify parts
of your code that have not been tested. It is particularly useful in
assessing code quality, as well-tested code is typically considered to
be of higher quality.
 
**Key Features of gcovr:**
- Summarizes code coverage statistics line-by-line.
- Can produce various output formats, including console reports,
  detailed or nested HTML reports, XML reports, JSON reports, and
  more.
- Works in conjunction with GCC's coverage instrumentation by
  compiling the code with specific flags (`--coverage -g -O0`) to emit
  coverage data.
- After tests have been run, `gcovr` can be used to process the raw
  coverage files generated by GCC and produce human-readable reports.
- It can be used to identify untested parts of the program, which is
  crucial for determining if the code is ready for release.
 
**Getting Started with gcovr:**
1. Instrument your code by recompiling with GCC using the flags
   `--coverage -g -O0`.
2. Execute your test suite to generate raw coverage files (`.gcda` and
   `.gcno` files).
3. Invoke `gcovr` from your build directory (with `-r` option pointing
   to the project root if necessary) to process the coverage data and
   generate reports.
 
**Example Usage:**
- To print a summary report on the console, simply run `gcovr` without
  arguments.
- To generate detailed HTML reports, use `gcovr --html-details
  coverage.html`.
- For nested HTML reports that include per-directory summaries, use
  `gcovr --html-nested coverage.html`.
 
**Compiler Options for Coverage:** When compiling your code for
coverage analysis with GCC or Clang, the key flags are `-fprofile-arcs
-ftest-coverage -fPIC -O0`. These flags instruct the compiler to add
logic to your program to track how often each part of the code
executes and to produce the necessary metadata files for coverage
analysis.
 
**Running the Program:** After compiling with the coverage flags, you
run your program normally. This execution generates the coverage data
files that `gcovr` will later process to create coverage reports.
 
**Processing Coverage:** Once you have run your program and generated
the `.gcda` coverage data files, `gcovr` is used to analyze these
files and produce coverage reports. `gcovr` automates the invocation
of the underlying `gcov` or `llvm-cov` tools, making it easier to
obtain a quick overview of coverage.
 
**Choosing the Right gcov Executable:** If using multiple compilers or
versions, you may need to specify which `gcov` executable `gcovr`
should use with the `--gcov-executable` option.
 
`gcovr` is a versatile tool for managing and summarizing code coverage
data, and it is especially useful for projects looking to maintain
high code quality through comprehensive testing.


### Comparison of lcov and gcovr

`gcovr` and `lcov` are both tools used to generate reports from
coverage data collected by GCC's `gcov`. However, they are distinct
tools with different output formats and usage patterns. Here's a
comparison highlighting their relationship:
 
**gcovr:**
- Written in Python.
- Provides a command-line interface to produce various report formats,
  including text summaries, HTML, XML, JSON, CSV, and SonarQube.
- Can generate detailed HTML reports with or without source code.
- Offers additional features such as filtering and excluding files
  from coverage.
- Typically invoked directly after running tests to generate coverage
  reports.
- Automates the invocation of `gcov` or `llvm-cov` to process `.gcda`
  and `.gcno` files.
 
**lcov:**
- Written in Perl.
- Initially designed as a graphical front-end for `gcov`.
- Generates HTML reports with an optional graphical interface that
  shows code coverage with highlighted source code.
- Provides features to add, combine, and remove coverage data files.
- Often used in conjunction with `genhtml`, another tool that
  processes `lcov`'s output to produce HTML reports.
- Can be part of more complex build systems and continuous integration
  workflows.
 
**Relationship:**
- Both `gcovr` and `lcov` serve the same purpose: to produce
  human-readable reports from the coverage data generated by `gcov`.
- They both read the same raw coverage data files created by
  instrumented binaries (`.gcda` and `.gcno` files).
- Users may choose one over the other based on their preference for
  report format, ease of use, or integration into existing tools and
  workflows.
- While `lcov` is specific to generating HTML reports (and uses
  `genhtml` for this purpose), `gcovr` is more flexible in terms of
  output formats.
- They are independent tools, not relying on each other, but they
  complement the ecosystem of code coverage analysis by offering
  different capabilities and focuses.
 
### What coverage files are produced by gcc?

The raw coverage files generated by GCC when using the `gcov` coverage
analysis tool are called:
 
- `.gcno` files (output by compiler): These are the "notes" files that contain information
  about the structure of the code. They are generated when the source
  files are compiled with the `-ftest-coverage` flag (or the
  `--coverage` flag which includes both `-ftest-coverage` and
  `-fprofile-arcs`).
 
- `.gcda` files (output by unit tests): These are the "data" files that contain the coverage
  counters and are created when the instrumented program is
  executed. They are generated as a result of running the compiled
  program with coverage instrumentation enabled (again, using
  `-fprofile-arcs` or `--coverage`).

- For my code the `unit_tests` binary creates these files as one can
  see with strace:

```
~/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b $ strace -f ./unit_tests 2>&1|grep \\.gc
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/_deps/googletest-build/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.gcda", O_RDWR|O_CREAT, 0666) = 3
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/_deps/googletest-build/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.gcda", O_RDWR|O_CREAT, 0666) = 3
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/CMakeFiles/unit_tests.dir/tests/test_DiagramBase.cpp.gcda", O_RDWR|O_CREAT, 0666) = 3
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/CMakeFiles/unit_tests.dir/src/DiagramBase.cpp.gcda", O_RDWR|O_CREAT, 0666) = 3
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/CMakeFiles/unit_tests.dir/src/CpuAffinityManagerBase.cpp.gcda", O_RDWR|O_CREAT, 0666) = 3
openat(AT_FDCWD, "/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/b/CMakeFiles/unit_tests.dir/tests/test_CpuAffinityManagerBase.cpp.gcda", O_RDWR|O_CREAT, 0666) = 3

```
 
- I also verified that `ctest` runs the unit_tests binary and writes
  the *.gcda files.
