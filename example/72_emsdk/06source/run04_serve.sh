export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/06"source/"

cd $ODIR/be
touch favicon.ico

python -m http.server --bind `hostname -I | awk '{print $1}'`

