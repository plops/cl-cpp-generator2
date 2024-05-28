// based on https://gist.github.com/austinmarton/2862515

#include <arpa/inet.h>
#include <iostream>
#include <linux/if_packet.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <span>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>

void check(int result, const std::string &msg) {
  if (-1 == result) {
    throw std::system_error(errno, std::generic_category(), msg);
  }
}

int main(int argc, char **argv) {
  try {
    const auto *ifName{"eth0"};
    auto sockfd{socket(PF_PACKET, SOCK_RAW, htons(0x800))};
    check(sockfd, "Failed  to create socket");
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
