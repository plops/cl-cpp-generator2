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
