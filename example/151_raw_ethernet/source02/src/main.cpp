// based on https://gist.github.com/austinmarton/2862515

#include <arpa/inet.h>
#include <array>
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
#include <unistd.h>
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
    // DGRAM .. don't merge packets together

    auto sockfd{socket(PF_PACKET, SOCK_DGRAM, htons(ETH_P_ALL))};
    // block release timeout 10ms

    auto packet_size{512U};
    auto frame_size{packet_size};
    auto frame_nr{2048U};
    auto block_nr{2U};
    auto req{tpacket_req3{.tp_block_size = ((frame_size * frame_nr) / block_nr),
                          .tp_block_nr = block_nr,
                          .tp_frame_size = frame_size,
                          .tp_frame_nr = frame_nr,
                          .tp_retire_blk_tov = 10,
                          .tp_sizeof_priv = 0,
                          .tp_feature_req_word = TP_FT_REQ_FILL_RXHASH}};
    setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));
    auto ll{
        sockaddr_ll({.sll_family = AF_PACKET,
                     .sll_protocol = htons(ETH_P_ALL),
                     .sll_ifindex = static_cast<int>(if_nametoindex("wlan0")),
                     .sll_hatype = ARPHRD_ETHER,
                     .sll_pkttype = PACKET_BROADCAST,
                     .sll_halen = 0})};
    bind(sockfd, reinterpret_cast<sockaddr *>(&ll), sizeof(ll));
    auto ring_info{tpacket_req3()};
    auto len{static_cast<socklen_t>(sizeof(ring_info))};
    getsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &ring_info, &len);
    auto ring_buffer{static_cast<char *>(
        mmap(0, ring_info.tp_block_size * ring_info.tp_block_nr,
             PROT_READ || PROT_WRITE, MAP_SHARED || MAP_LOCKED, sockfd, 0))};
    auto current_block{0};
    // packet processing loop

    while (true) {
      auto pfd{pollfd({{.fd = sockfd, .events = POLLIN}})};
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
          }
        }
      }
    }
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
