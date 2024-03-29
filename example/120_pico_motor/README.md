
- video how to use https://www.youtube.com/watch?v=HDxrrG1RFaw

- pico needs 100mA @ 5V

- motor needs 250mA @ 5V (might also run at lower voltage ~3V)

- 2048 steps per revolution
- 2ms per step

- stepper library for arduino for uln2003

- focus https://www.youtube.com/watch?v=e3WctzSnUvc
-   https://www.youtube.com/watch?v=VM3S9CiyPzY 28BYJ-48 Stepper Motor
  with Raspberry PI Pico (C++ and FreeRTOS)
  -https://github.com/jondurrant/RPIPicoStepperExp
  - slot detector to build an encoder

- how to use both cores
  https://www.youtube.com/watch?v=nD8XeWjn-2w

- drive two stepper motors using the pico's PIO state machines
  https://vanhunteradams.com/Pico/Steppers/Lorenz.html
  - https://www.youtube.com/watch?v=IuZq3p86Ydg Raspberry Pi Pico
    Lecture 25: PIO Stepper Motor Driver
  - explains half stepping
  - number of steps, speed, direction
  - https://github.com/vha3/Hunter-Adams-RP2040-Demos/tree/master/Stepper_Motors/Stepper_Speed_Control
  - he has several versions of this code. i'm most interested in
    position only control with 2 state machines
    
- sdk doc https://datasheets.raspberrypi.com/pico/raspberry-pi-pico-c-sdk.pdf

- windows installer https://github.com/raspberrypi/pico-setup-windows/releases/

- install on linux

- debian:

```
sudo apt install cmake \
  gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib
```

- arch linux:
```

sudo pacman -S \
arm-none-eabi-newlib \
arm-none-eabi-gcc 


cd ~/src
git clone --recurse-submodules -j8  https://github.com/raspberrypi/pico-sdk

# takes 3.6GB


cp /home/martin/src/pico-sdk/external/pico_sdk_import.cmake \
/home/martin/stage/cl-cpp-generator2/example/120_pico_motor/source01/

```


- build with the sdk

```

cd /home/martin/stage/cl-cpp-generator2/example/120_pico_motor/source01/
mkdir b
cd b
cmake .. -G Ninja -DPICO_SDK_PATH=/home/martin/src/pico-sdk

# install hello_world.uf2 via drag and drop or copy
# plug pico pi into usb port
sudo mount /dev/sda1 /mnt
sudo cp hello_world.uf2 /mnt/

# show serial port (usb)

sudo screen /dev/ttyACM0 115200

```

- state machine simulator
https://rp2040pio-docs.readthedocs.io/en/latest/introduction.html
 - i have compiled it but did not try it
 
- pulley and belt
https://youtu.be/0sKqc5IwQQQ?t=648


- uart protocol options
- https://github.com/bmellink/IBusBM 
  - 140 stars
  - flysky rc protocol
  
- https://github.com/MightyPork/TinyFrame
  - 266 stars
  - C99
  - 
  
-  https://github.com/MaJerle/lwow
  - 133 stars
  - C99
  
-  https://github.com/lexus2k/tinyproto
  - 180 stars
  - layer 2
  - C, C++
  - i like this most, protocol buffers can be used on top
  - based on rfc 1662  PPP in HDLC-like Framing
  - https://www.rfc-editor.org/rfc/rfc1662 
- https://github.com/lexus2k/tinyslip
 - 2 stars
 - 3 years old
 - rfc 1055 "A NONSTANDARD FOR TRANSMISSION OF IP DATAGRAMS OVER SERIAL LINES: SLIP"
 - https://www.rfc-editor.org/rfc/rfc1055.html
 - no error detection
- https://embeddedproto.com/
  - C++
  - only free for non-commercial use

# what is the best way to communicate from pico to pc

You cannot use a USB CDC serial connection during debugging. 

https://github.com/hathach/tinyusb


hitec servo https://www.youtube.com/watch?v=iBbFxsB6KV0
