| file  | state | lang | comment                                                             |
|-------|-------|------|---------------------------------------------------------------------|
| gen01 |       | C++  | read ADC data using soapysdr                                        |
| gen02 |       | C++  | GUI with imgui bundle (sdr control, recording, realtime processing) |
| gen03 |       | C++  | GUI to decode GPS of recording                                      |
|       |       |      |                                                                     |


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

- http://www.taroz.net/GNSS-Radar.html
- http://www.taroz.net/img/data/SDR_SummerSchool_2014.pdf
- http://www.aholme.co.uk/GPS/Main.htm Homemade GPS Receiver
  - sample 40000 points at 10MHz, rotate in Fourierspace by +/- 20
    bins to find doppler shift
- MIT Licensed receiver:
  https://github.com/hamsternz/Full_Stack_GPS_Receiver/blob/master/fast_fsgps/gold_codes.c

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
