#include <ISocket.h>
#include <TCPSocket.h>
#include <memory>

using namespace std;

int main(int argc, char *argv[]) {
  // auto socket = make_unique<ISocket>{nullptr};
  unique_ptr<ISocket> socket;
  if (argc < 2) {
    perror("Too few arguments");
    return EXIT_FAILURE;
  }
  string protocolChoice = argv[1];
  if (protocolChoice == "tcp") {
    socket = make_unique<TCPSocket>{nullptr};
  }

  return 0;
}
