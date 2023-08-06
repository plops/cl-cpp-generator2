| file     | state | lang | comment                                                             |
|----------|-------|------|---------------------------------------------------------------------|
| gen01    |       | C++  | read ADC data using soapysdr                                        |
| gen02    |       | C++  | GUI with imgui bundle (sdr control, recording, realtime processing) |
| gen03    |       | C++  | GUI to decode GPS of recording                                      |
| source04 |       | C    | reference PRN from http://www.jks.com/gps/SearchFFT.cpp             |
|          |       |      |                                                                     |


- https://youtu.be/ANry8tAPjN4?t=1799
```
cat << EOF >>  /etc/portage/package.accept_keywords/package.accept_keywords
net-wireless/sdrplay ~amd64
net-wireless/soapysdr ~amd64
net-wireless/soapysdrplay ~amd64
net-wireless/soapyplutosdr ~amd64
net-libs/libiio ~amd64
net-libs/libad9361-iio ~amd64
EOF

cat << EOF >>  /etc/portage/package.use/package.use
net-wireless/soapysdr -bladerf -hackrf plutosdr python -rtlsdr -uhd

EOF

MAKEOPTS=-j12 sudo emerge -av sdrplay net-wireless/soapysdr soapysdrplay


sudo /usr/bin/sdrplay_apiService 
SoapySDRUtil --find

```

- https://github.com/pothosware/SoapySDR/wiki/Cpp_API_Example


- command line parser

```

cd ~/src
git clone https://github.com/CLIUtils/CLI11

wget https://github.com/CLIUtils/CLI11/releases/download/v2.3.2/CLI11.hpp

wget https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp

```

- gui

- examples of wxwidgets https://github.com/gammasoft71/Examples_wxWidgets
- bundle for imgui https://github.com/pthom/imgui_bundle#install-for-c
  - _example_integration shows how to use imgui_bundle in your own project
  - no need to add imgui_bundle as a submodule or install it, cmake
    will fetch it
  - this fetches a 550MB of dependencies
  - after release build (154 files) cmake-build-release/ contains
    577MB (27MB more)
  - after debug build the size of _deps is 662MB the rest is 4MB
  - i don't want to download the dependencies twice. try to reuse
    dependencies from cmake-build-release/ in cmake-build-debug/.
    - disable release build in clion
	- create debug directory and create link
```
mkdir -p /home/martin/stage/cl-cpp-generator2/example/131_sdr/source02/cmake-build-debug
cp -ar /home/martin/stage/cl-cpp-generator2/example/131_sdr/source02/cmake-build-release/_deps /home/martin/stage/cl-cpp-generator2/example/131_sdr/source02/cmake-build-debug
cp -ar /home/martin/stage/cl-cpp-generator2/example/131_sdr/source02/cmake-build-release/_deps /home/martin/stage/cl-cpp-generator2/example/131_sdr/source02/_deps


```

    - enable debug build in clion

    - i have to delete the CMakeCache.txt file inside the _deps folder. then i can speed up the running download by overwriting the _deps folder
	
	
```
martin@archlinux ~/stage/cl-cpp-generator2/example/131_sdr/source02/_deps $ find .|grep CMakeCache|xargs rm
martin@archlinux ~/stage/cl-cpp-generator2/example/131_sdr/source02 $ cp -ar _deps/ cmake-build-release/
martin@archlinux ~/stage/cl-cpp-generator2/example/131_sdr/source02 $ ./setup02_get_assets.sh

```
- make sure to copy the assets into the new folder as well (roboto font) before running the release build


# GPS

- http://www.taroz.net/GNSS-Radar.html which gps satellites are currently visible 
- http://www.taroz.net/img/data/SDR_SummerSchool_2014.pdf
- http://www.aholme.co.uk/GPS/Main.htm Homemade GPS Receiver
  - sample 40000 points at 10MHz, rotate in Fourierspace by +/- 20
    bins to find doppler shift
- MIT Licensed receiver:
  https://github.com/hamsternz/Full_Stack_GPS_Receiver/blob/master/fast_fsgps/gold_codes.c
- http://www.jks.com/gps/gps.html
- http://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle satellite orbit elements


# gen03

- pull in minimal code for implot
- implot.h, implot_internal.h, implot.cpp, implot_items.cpp

```

for i in implot.h implot_internal.h implot.cpp implot_items.cpp; do
wget https://raw.githubusercontent.com/epezent/implot/master/$i
done



```

- imgui dependencies:
   - Add all source files in the root folder: imgui/{*.cpp,*.h}.
   - Add selected imgui/backends/imgui_impl_xxxx{.cpp,.h} files corresponding to the technology you use from the imgui/backends/ folder
   - std::string users: Add misc/cpp/imgui_stdlib.* to easily use InputText with std::string.
```

for i in \
  imconfig.h \
  imgui.h imgui.cpp imgui_draw.cpp imgui_internal.h imgui_tables.cpp \
  imgui_widgets.cpp imstb_rectpack.h imstb_textedit.h imstb_truetype.h \
  backends/imgui_impl_glfw.h backends/imgui_impl_opengl3.h \
  backends/imgui_impl_opengl3.cpp backends/imgui_impl_opengl3_loader.h \
  misc/cpp/imgui_stdlib.cpp misc/cpp/imgui_stdlib.h ;do
wget  https://raw.githubusercontent.com/ocornut/imgui/master/$i
done


```


# Book: El Rabani: Introduction to GPS (2002)

- p 54. Figure 4.1 depicts GPS signal vs thermal noise power at 2MHz
  bandwidth and 290K

Received GNSS signals are right-hand circularly polarized (RHCP), and
the antenna should be designed as such.

When the GNSS signal is reﬂected off an object, an undesirable
situation for a system attempting to measure time- of-ﬂight, the
polarization will ﬂip to left-hand circular polarization (LHCP). An
RHCP antenna is quite effective in suppressing the LHCP reﬂection and
mini- mizing this error source.

- in figure 4.6 he describes a noticable peak of the gps signal,
  perhaps 4dB above the noise


# Example data

```
  54M Jul 30 15:03 gps.samples.1bit.I.fs5456.if4092.bin
 213M Aug  1 19:09 gps.samples.cs16.fs5456.if4092.dat

```
- converted from 1-bit I samples to complex<short>, so that I can load
  it into my program

```
/home/martin/src/gps/cmake-build-debug/imgui_dsp
PRN Nav Doppler   Phase   MaxSNR
1    63    1375    198.2   322.1    ********************************
2    56   -4000    383.4    18.6    *
3    37    4375    475.1    11.7    *
4    35   -3875    497.4    13.6    *
5    64     875    879.0    38.6    ***
6    36    2875     18.0    14.5    *
7    62   -2250    183.0    15.3    *
8    44   -3125    501.0    15.3    *
9    33   -4625    480.2    13.4    *
10   38   -1500    336.4    37.1    ***
11   46    3875    266.8    13.0    *
12   59    4875    804.0    14.7    *
13   43    1125    840.6    50.0    ****
14   49    2875    941.8    15.3    *
15   60    3125    532.9    13.5    *
16   51    2250    318.9    35.2    ***
17   57   -2250     64.1    11.6    *
18   50    3500     29.1    12.7    *
19   54   -2250    205.1    14.6    *
20   47    4125    853.3    11.7    *
21   52    2125    962.4   102.9    **********
22   53    4875    792.0    11.4    *
23   55    -625     50.4    40.9    ****
24   23   -4250    146.8    12.4    *
25   24   -3750    435.9    79.7    *******
26   26    -125    353.8    13.1    *
27   27    1375    100.5    13.9    *
28   48   -4500     82.7    12.4    *
29   61   -2125     39.9   197.3    *******************
30   39   -2125    857.8   269.9    **************************
31   58   -1875    534.4   202.7    ********************
32   22    2125    765.6    12.8    *

Process finished with exit code 0

```


# Power supply via Bias-Tee


- above 5V the amplifier seems to become unstable
- even normal USB voltage makes it unstable
- on a programmable powersupply i saw 5V and 17mA

- on laptop i saw occasionally this:
```
[15161.065417] usb usb2-port2: over-current condition

```

- i now have a 3.3v buck boost convert (i think) 
# speed up C++ compilation

- https://devtalk.blender.org/t/speed-up-c-compilation/30508
- https://news.ycombinator.com/item?id=37014842

```

sudo emerge -av mold ccache


```

## forward declaration

 Some standard types have forward declarations in #include <iosfwd>, but not all.
 
## unity build

- the sequence of source files matters
- name clashes between static variables can be problematic
- you may want to create build commands json with unity disabled

```
set(CMAKE_UNITY_BUILD_BATCH_SIZE 10)

# enable Unity build
set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)


# exclude main.cpp from Unity build (this is useful if main.cpp is the only file that I edit right now)
set_source_files_properties(src/main.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)



 
 ```
## include-what-you-use

```
  mkdir iwyubuild && cd iwyubuild
  CC="clang" CXX="clang++" cmake -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=include-what-you-use ..

```

- alternatively an older solution is to use the compilation database

```
 mkdir build && cd build
  CC="clang" CXX="clang++" cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
  iwyu_tool.py -p .

source03/cmake-build-release $ iwyu_tool.py -p . ../src/main.cpp 

```
