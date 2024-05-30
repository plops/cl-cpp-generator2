#include <arpa/inet.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <linux/if_packet.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <poll.h>
#include <span>
#include <sys/mman.h>
#include <sys/socket.h>
#include <system_error>
#include <thread>
#include <unistd.h>
// Note: not working, yet

// assume we receive a packet for each line of a video camera
// i didn't add error handling. i suggest strace instead

class VideoLine {
public:
  uint16_t width;
  uint16_t height;
  uint32_t timestamp;
  std::array<uint8_t, 320> imageData;
};

int main(int argc, char **argv) {
  std::cout << ""
            << " argc='" << argc << "' "
            << " argv[0]='" << argv[0] << "' " << std::endl;
  try {
    // SOCK_STREAM .. merges packets together
    // SOCK_DGRAM  .. don't merge packets together
    // SOCK_RAW    .. capture ethernet header as well

    auto sockfd{socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))};
    // bind socket to the hardware interface

    auto ifindex{static_cast<int>(if_nametoindex("lo"))};
    auto ll{sockaddr_ll({.sll_family = AF_PACKET,
                         .sll_protocol = htons(ETH_P_ALL),
                         .sll_ifindex = ifindex})};
    std::cout << ""
              << " ifindex='" << ifindex << "' " << std::endl;
    bind(sockfd, reinterpret_cast<sockaddr *>(&ll), sizeof(ll));
    // define version

    auto version{TPACKET_V2};
    setsockopt(sockfd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version));
    // configure ring buffer

    auto block_size{static_cast<uint32_t>(2 * getpagesize())};
    auto block_nr{2U};
    auto frame_size{2048U};
    auto frame_nr{(block_size / frame_size) * block_nr};
    auto req{tpacket_req{.tp_block_size = block_size,
                         .tp_block_nr = block_nr,
                         .tp_frame_size = frame_size,
                         .tp_frame_nr = frame_nr}};
    setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));
    // map the ring buffer

    auto mmap_size{block_size * block_nr};
    auto mmap_base{
        mmap(0, mmap_size, PROT_READ || PROT_WRITE, MAP_SHARED, sockfd, 0)};
    auto rx_buffer_size{block_size * block_nr};
    auto rx_buffer_addr{mmap_base};
    auto rx_buffer_idx{0};
    auto rx_buffer_cnt{(block_size * block_nr) / frame_size};
    auto base{rx_buffer_addr + rx_buffer_idx * frame_size};
    volatile tpacket2_hdr *header{static_cast<tpacket2_hdr *>(base)};
    auto pollfds{pollfd({.fd = sockfd, .events = POLLIN, .revents = 0})};
    auto poll_res{ppoll(&pollfds, 1, nullptr, nullptr)};
    if ((POLLIN & pollfds.revents)) {
      auto data{base + header->tp_net};
      auto data_len{header->tp_snaplen};
      auto ts{timespec({.tv_sec = header->tp_sec, .tv_nsec = header->tp_nsec})};
      std::cout << ""
                << " ts.tv_sec='" << ts.tv_sec << "' "
                << " ts.tv_nsec='" << ts.tv_nsec << "' "
                << " data_len='" << data_len << "' " << std::endl;
      // hand frame back to kernel

      header->tp_status = TP_STATUS_KERNEL;
    }
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
