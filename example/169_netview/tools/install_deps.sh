cd ~/src
tar xaf /mnt3/var/cache/distfiles/ffmpeg-6.1.2.tar.xz
cd ~/src/ffmpeg-6.1.2
sh ~/stage/cl-cpp-generator2/example/169_netview/tools/configure_ffmpeg.sh

cd ~/src
git clone /mnt4/persistent/lower/home/martin/src/avcpp

cd ~/src
git clone /mnt4/persistent/lower/home/martin/src/capnproto

cd capnproto
mkdir b
cd b
cmake .. \
-G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/martin/vulkan \
-DBUILD_TESTING=OFF

ninja
ninja install