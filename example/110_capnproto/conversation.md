Q: Show modern C++ code for a capnproto server.

A: Here's an example of modern C++ code for a Cap'n Proto server using
the Cap'n Proto C++ library:

```c++

#include <capnp/rpc.h>
#include <capnp/rpc-twoparty.h>
#include <kj/async.h>
#include "my_service.capnp.h"

class MyServiceImpl final : public MyService::Server {
public:
  kj::Promise<void> calculate(MyService::CalculateContext context) override {
    int arg1 = context.getParams().getArg1();
    int arg2 = context.getParams().getArg2();

    // Perform calculation...
    int result = arg1 + arg2;

    // Build response.
    auto response = context.getResults();
    response.setResult(result);

    return kj::READY_NOW;
  }
};

int main(int argc, char* argv[]) {
  kj::setupAsyncIo();

  MyServiceImpl myService;

  // Create Cap'n Proto server.
  kj::WaitScope waitScope;
  capnp::TwoPartyServer server(kj::heap<capnp::RpcSystem>(myService),
                               argc > 1 ? argv[1] : "127.0.0.1:12345");
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}
```

In this example, we define a MyServiceImpl class that implements the
MyService interface, which is defined in the my_service.capnp Cap'n
Proto schema file. The MyService interface contains a single method
called calculate that takes two integer arguments and returns an
integer result.

In the calculate method, we extract the arguments from the incoming
request, perform the calculation, and build a response with the
result.

In the main function, we create an instance of MyServiceImpl and pass
it to a capnp::RpcSystem, which handles incoming requests and forwards
them to the appropriate service method. We then create a
capnp::TwoPartyServer and pass the RpcSystem to it, along with a
network address on which to listen for incoming connections. Finally,
we enter an infinite loop that waits for incoming connections.


Q: Show the service protocol definition file and the cmake file

A: Sure, here's an example of the Cap'n Proto service protocol
definition file and a CMake file that can be used to build the server:

The Cap'n Proto service protocol definition file (my_service.capnp):

```ruby

@0xdc35abf53c82d81e;

interface MyService {
  struct Params {
    arg1 @0 :Int32;
    arg2 @1 :Int32;
  }

  struct Results {
    result @0 :Int32;
  }

  calculate (params :Params) -> (results :Results);
}
```

The CMake file (CMakeLists.txt):

```scss

cmake_minimum_required(VERSION 3.12)
project(my_service)

set(CMAKE_CXX_STANDARD 17)

find_package(CapnProto CONFIG REQUIRED)

add_executable(my_service
  main.cpp
  my_service.capnp.c++
)

target_link_libraries(my_service
  CapnProto::capnp-rpc
  CapnProto::capnp
)

target_include_directories(my_service PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/my_service.capnp.c++
  COMMAND capnp compile -oc++ ${CMAKE_CURRENT_SOURCE_DIR}/my_service.capnp -I${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS my_service.capnp
)

add_custom_target(my_service_capnp DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/my_service.capnp.c++)

add_dependencies(my_service my_service_capnp)
```

In this example, we define a CMake project called my_service and set
the C++ standard to 17. We then use the find_package command to locate
the Cap'n Proto library.

We create an executable called my_service and add the source files
main.cpp and my_service.capnp.c++. We link the capnp-rpc and capnp
libraries to the executable and add the CMAKE_CURRENT_BINARY_DIR
directory to the include path.

We then use the add_custom_command and add_custom_target commands to
generate the my_service.capnp.c++ file from the my_service.capnp
protocol definition file using the capnp compiler. Finally, we add a
dependency on the my_service_capnp target to the my_service target so
that the protocol definition file is compiled before the executable is
built.