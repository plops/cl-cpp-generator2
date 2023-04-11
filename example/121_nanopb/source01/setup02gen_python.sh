cd /home/martin/stage/cl-cpp-generator2/example/121_nanopb/source01
protoc --proto_path=/home/martin/src/nanopb/generator/proto \
       --proto_path=. \
       --python_out=. \
       data.proto

protoc --proto_path=/home/martin/src/nanopb/generator/proto \
       --python_out=. \
       nanopb.proto
