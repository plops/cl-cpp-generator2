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
    auto ring_info{tpacket_req3()};
    auto len{static_cast<socklen_t>(sizeof(ring_info))};
    getsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &ring_info, &len);
    auto ring_buffer{static_cast<char *>(
        mmap(0, ring_info.tp_block_size * ring_info.tp_block_nr,
             PROT_READ || PROT_WRITE, MAP_SHARED || MAP_LOCKED, sockfd, 0))};
    auto current_block{0};
    // packet processing loop

    while (true) {
      auto pfd{pollfd({.fd = sockfd, .events = POLLIN})};
      auto pollresult{poll(&pfd, 1, -1)};
      if (0 < pollresult) {
        for (auto frame_idx = 0;
             frame_idx < static_cast<int>(ring_info.tp_frame_nr);
             frame_idx += 1) {
          auto hdr{reinterpret_cast<tpacket3_hdr *>(
              ring_buffer + current_block * ring_info.tp_block_size +
              frame_idx * ring_info.tp_frame_size)};
          if ((TP_STATUS_USER & hdr->hv1.tp_rxhash)) {
            auto placement_address{reinterpret_cast<char *>(hdr) +
                                   hdr->tp_next_offset};
            auto videoLine{new (placement_address) VideoLine()};
            std::cout << ""
                      << " videoLine->width='" << videoLine->width << "' "
                      << std::endl;
            hdr->hv1.tp_rxhash = 0;
            // delete of videoLine not required as it is placement new and
            // memory is in ring buffer
          }
        }
        // move to next block

        current_block = ((current_block + 1) % ring_info.tp_block_nr);
      }
      // prevent busy wait

      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
