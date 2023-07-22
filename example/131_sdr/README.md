
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

```