// based on https://gist.github.com/austinmarton/2862515

#include <array>
#include <cstring>
#include <format>
#include <iostream>
#include <linux/if_packet.h>
#include <net/if.h>
#include <poll.h>
#include <span>
#include <sys/mman.h>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>
// assume we receive a packet for each line of a video camera

class VideoLine {
public:
  uint16_t width;
  uint16_t height;
  uint32_t timestamp;
  std::array<uint8_t, 320> imageData;
};

void check(int result, const std::string &msg) {
  if (-1 == result) {
    throw std::system_error(errno, std::generic_category(), msg);
  }
}

int main(int argc, char **argv) {
  try {
    // don't merge packets together

    auto sockfd{socket(PF_PACKET, SOCK_DGRAM, htons(ETH_P_ALL))};
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
