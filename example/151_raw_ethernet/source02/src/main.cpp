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

    auto packet_size{512};
    auto frame_size{packet_size};
    auto frame_nr{2048};
    auto block_nr{2};
    auto req{tpacket_req3{.tp_block_size = ((frame_size * frame_nr) / block_nr),
                          .tp_block_nr = block_nr,
                          .tp_frame_size = frame_size,
                          .tp_frame_nr = frame_nr,
                          .tp_retire_blk_tov = 10,
                          .tp_sizeof_priv = 0,
                          .tp_feature_req_word = TP_FT_REQ_FILL_RXHASH}};
    setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
