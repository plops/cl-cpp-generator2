protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` glgui.proto
protoc --cpp_out=. glgui.proto
