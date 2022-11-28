mkdir ber
cd ber
source "/home/martin/src/emsdk/emsdk_env.sh"
emcmake cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -v
# sudo dnf install binaryen
mv index.wasm index0.wasm
wasm-opt index0.wasm -o index.wasm --fast-math -O --intrinsic-lowering -O --vacuum --strip-debug
