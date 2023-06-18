```
sudo pacman -S opencv vtk hdf5
```
- install google fruit
```
https://github.com/google/fruit/archive/refs/tags/v3.7.1.tar.gz
 cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/home/martin/fruit
 
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


# Ask GPT4

- i ask chatgpt how to structure the project

- Q: I learned that you can't get the opencv code right on first
try. Considering this I suggest to break the tasks down and do some
spiking with the goal to write small test progams for individual
functionality. First without google fruit. Once the test programs work
classes with google fruit will be developed for the final phase. Make
a list of test programs that need to be developed and order them in
the sequence we should tackle them.  ChatGPT

- A: That's a very prudent approach, breaking down the tasks and
understanding them first before trying to assemble everything
together. Here's a list of small test programs that you should try to
write in sequence:

1. Display an Image: Create a simple program that can read an image
   file and display it using the HighGUI window. This ensures that you
   understand how to read and display images, which is essential for
   visualization.

2. Generate an ArUco Marker: Write a program to generate an ArUco
   marker and save it as an image file. This will help you understand
   how to work with ArUco markers in OpenCV.

3. Display an ArUco Marker: Enhance the first program to display the
   generated ArUco marker. This will help you understand how to use
   the markers with HighGUI window.

4. Detect ArUco Marker: Write a program to detect the ArUco marker
   from an image. This will ensure that you understand how to process
   images and detect markers.

5. Capture from Webcam: Write a program that captures the video feed
   from the webcam and display it in a HighGUI window. This will make
   sure you understand video capture and display.

6. Detect ArUco Marker in Video: Combine the webcam capture and ArUco
   detection code to detect ArUco markers in the video feed from the
   webcam.

7. Generate a Charuco Board: Create a program that can generate a
   Charuco board and display it on the screen. This is a first step to
   calibrating the camera.

8. Detect Charuco Board: Write a program that can detect the Charuco
   board from an image or video feed. This will help you understand
   how to detect complex patterns.

9. Calibrate Camera using Charuco Board: Write a program that uses the
   detected Charuco board to calibrate the camera. This will be a
   crucial part of your final program.

Once you've written and understand all these programs, you can then
start integrating them into a class-based system using Google
Fruit. This approach will ensure that you understand each component of
your final program and can debug them individually before trying to
assemble everything together.
