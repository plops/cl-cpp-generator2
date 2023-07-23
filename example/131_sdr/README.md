| file  | state | lang | comment                      |
|-------|-------|------|------------------------------|
| gen01 |       | C++  | read ADC data using soapysdr |
| gen02 |       | C++  | GUI with imgui bundle        |
|       |       |      |                              |


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

