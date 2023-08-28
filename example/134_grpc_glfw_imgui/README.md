- minimal example of imgui with grpc

```
sudo emerge -av net-libs/grpc dev-libs/protobuf

```

- compile proto file

```

protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` glgui.proto


```