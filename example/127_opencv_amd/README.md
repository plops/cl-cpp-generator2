```
sudo pacman -S opencv vtk hdf5
```
- install google fruit
```
https://github.com/google/fruit/archive/refs/tags/v3.7.1.tar.gz
 cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
```
- compile opencv from source
```
cd ~/src
wget https://github.com/opencv/opencv/archive/4.7.0.zip
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz
unzip 4.7.0.zip

mkdir opencv_build
cd opencv_build

cmake \
-G Ninja \
-DBUILD_JAVA=OFF \
-DENABLE_FAST_MATH=ON \
-DOPENCV_ENABLE_NONFREE=ON \
-DWITH_OPENGL=OFF \
-DCMAKE_INSTALL_PREFIX=/home/martin/opencv \
-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.7.0/modules ../opencv-4.7.0

ccmake ../opencv-4.7.0

ninja
ninja install
```

- the code that gpt4 calls doesn't seem to work (drawCharucoBoard)
- in the example 72_emsdk i already tried something similar
- aruco seems to be in opencv_contrib
