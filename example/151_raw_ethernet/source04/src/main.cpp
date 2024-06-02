#include "PacketReceiver.h"
#include <iostream>
#include <system_error>

int main(int argc, char **argv) {
  std::cout << ""
            << " argc='" << argc << "' "
            << " argv[0]='" << argv[0] << "' " << std::endl;
  try {
    auto cb{[](const uint8_t *data, size_t size) {
      if (0 < size) {
        std::cout << static_cast<int>(data[0]) << std::endl;
      }
    }};
    auto r{PacketReceiver(cb)};
    r.receive();
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  // unreachable:

  return 0;
}
