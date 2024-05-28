// based on https://gist.github.com/austinmarton/2862515

#include <arpa/inet.h>
#include <array>
#include <cstring>
#include <iostream>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
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
    // Open PF_PACKET socket

    auto ifName{"lo"};
    auto sockfd{socket(PF_PACKET, SOCK_RAW, htons(0x800))};
    check(sockfd, "Failed to create socket");
    // Set interface to promiscuous mode

    auto ifopts{ifreq{}};
    strncpy(ifopts.ifr_name, ifName, IFNAMSIZ - 1);
    check(ioctl(sockfd, SIOCGIFFLAGS, &ifopts),
          "Failed to get interface flags");
    ifopts.ifr_flags = ifopts.ifr_flags || IFF_PROMISC;
    check(ioctl(sockfd, SIOCSIFFLAGS, &ifopts),
          "Failed to set interface flags");
    // Allow the socket to be reused

    auto sockopt{SO_REUSEADDR};
    check(
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &sockopt, sizeof(sockopt)),
        "Failed to set socket option SO_REUSEADDR");
    // Bind to device

    check(setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, ifName, IFNAMSIZ - 1),
          "Failed to set socket option SO_BINDTODEVICE");
    // Receive loop

    while (true) {
      auto buf{std::array<uint8_t, 1024>{}};
      auto numbytes{
          recvfrom(sockfd, buf.data(), buf.size(), 0, nullptr, nullptr)};
      check(numbytes, "Failed to receive data");
      auto eh{reinterpret_cast<ether_header *>(buf.data())};
      auto receivedMac{std::span<const uint8_t, 6>(eh->ether_dhost, 6)};
    }
    close(sockfd);
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
