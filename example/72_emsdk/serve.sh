export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$1"source/"

cd $ODIR/b
touch favicon.ico

python -m http.server

