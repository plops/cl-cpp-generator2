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
unzip 4.7.0.zip
cd opencv-4.7.0
mkdir b
cd b
cmake .. -G Ninja
ccmake ..
cmake .. -G Ninja -DBUILD_JAVA=OFF \
-DENABLE_FAST_MATH=ON \
-DOPENCV_ENABLE_NONFREE=ON \
-DWITH_OPENGL=ON 
```

- the code that gpt4 calls doesn't seem to work (drawCharucoBoard)
