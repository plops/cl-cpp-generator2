# keep bootset pressed and reconnect usb-a

cmake .. -G Ninja -DPICO_SDK_PATH=/home/martin/src/pico-sdk
ninja
sudo mount /dev/sda1 /mnt
sudo cp hello_world.uf2 /mnt
sudo umount /mnt
