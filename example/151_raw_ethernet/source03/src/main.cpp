#include <arpa/inet.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <iomanip>
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
    auto sockfd{socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))};
    if (sockfd < 0) {
      std::cout << "error opening socket. try running as root" << std::endl;
      return -1;
    }
    // bind socket to the hardware interface

    auto ifindex{static_cast<int>(if_nametoindex("wlan0"))};
    auto ll{sockaddr_ll(
        {.sll_family = AF_PACKET,
         .sll_protocol = htons(ETH_P_ALL),
         .sll_ifindex = ifindex,
         .sll_hatype = ARPHRD_ETHER,
         .sll_pkttype = PACKET_HOST | PACKET_OTHERHOST | PACKET_BROADCAST,
         .sll_halen = 0,
         .sll_addr = {0, 0, 0, 0, 0, 0, 0, 0}})};
    std::cout << ""
              << " ifindex='" << ifindex << "' " << std::endl;
    bind(sockfd, reinterpret_cast<sockaddr *>(&ll), sizeof(ll));
    // define version

    auto version{TPACKET_V2};
    setsockopt(sockfd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version));
    // configure ring buffer

    auto block_size{static_cast<uint32_t>(1 * getpagesize())};
    auto block_nr{2U};
    auto frame_size{2048U};
    auto frame_nr{(block_size / frame_size) * block_nr};
    auto req{tpacket_req{.tp_block_size = block_size,
                         .tp_block_nr = block_nr,
                         .tp_frame_size = frame_size,
                         .tp_frame_nr = frame_nr}};
    std::cout << ""
              << " block_size='" << block_size << "' "
              << " block_nr='" << block_nr << "' "
              << " frame_size='" << frame_size << "' "
              << " frame_nr='" << frame_nr << "' " << std::endl;
    setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));
    // map the ring buffer

    auto mmap_size{block_size * block_nr};
    auto mmap_base{
        mmap(0, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, sockfd, 0)};
    auto rx_buffer_size{block_size * block_nr};
    auto rx_buffer_addr{mmap_base};
    auto rx_buffer_cnt{(block_size * block_nr) / frame_size};
    std::cout << ""
              << " rx_buffer_size='" << rx_buffer_size << "' "
              << " rx_buffer_cnt='" << rx_buffer_cnt << "' " << std::endl;
    auto idx{0U};
    auto old_arrival_time64{uint64_t(0)};
    while (true) {
      auto pollfds{pollfd({.fd = sockfd, .events = POLLIN, .revents = 0})};
      auto poll_res{ppoll(&pollfds, 1, nullptr, nullptr)};
      if (poll_res < 0) {
        std::cout << "error in ppoll"
                  << " poll_res='" << poll_res << "' "
                  << " errno='" << errno << "' " << std::endl;
      } else if (poll_res == 0) {
        std::cout << "timeout" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
      } else {
        if (POLLIN & pollfds.revents) {
          auto base{reinterpret_cast<uint8_t *>(rx_buffer_addr)};
          auto header{
              reinterpret_cast<tpacket2_hdr *>(base + idx * frame_size)};
          // Iterate through packets in the ring buffer

          do {
            if (header->tp_status & TP_STATUS_USER) {
              if (header->tp_status & TP_STATUS_COPY) {
                std::cout << "copy"
                          << " idx='" << idx << "' " << std::endl;
              } else if (header->tp_status & TP_STATUS_LOSING) {
                auto stats{tpacket_stats()};
                auto stats_size{static_cast<socklen_t>(sizeof(stats))};
                getsockopt(sockfd, SOL_PACKET, PACKET_STATISTICS, &stats,
                           &stats_size);
                std::cout << "loss"
                          << " idx='" << idx << "' "
                          << " stats.tp_drops='" << stats.tp_drops << "' "
                          << " stats.tp_packets='" << stats.tp_packets << "' "
                          << std::endl;
              }
              auto data{reinterpret_cast<uint8_t *>(header) + header->tp_net};
              auto data_len{header->tp_snaplen};
              auto arrival_time64{1000000000 * header->tp_sec +
                                  header->tp_nsec};
              auto delta64{arrival_time64 - old_arrival_time64};
              auto delta_ms{delta64 / 1.00e+6};
              auto arrival_timepoint{
                  std::chrono::system_clock::from_time_t(header->tp_sec) +
                  std::chrono::nanoseconds(header->tp_nsec)};
              auto time{
                  std::chrono::system_clock::to_time_t(arrival_timepoint)};
              auto local_time{std::localtime(&time)};
              auto local_time_hr{
                  std::put_time(local_time, "%Y-%m-%d %H:%M:%S")};
              std::cout << local_time_hr << " " << std::setfill(' ')
                        << std::setw(4 + 6) << std::fixed
                        << std::setprecision(6)
                        << (delta_ms < 1000 ? std::to_string(delta_ms)
                                            : "xxx.xxxxxx")
                        << " " << std::dec << std::setw(6) << data_len << " "
                        << std::setw(4) << idx << " ";
              for (unsigned int i = 0; i < (data_len < 64U ? data_len : 64U);
                   i += 1) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(data[i]) << " ";
                if (0 == (i % 8)) {
                  std::cout << " ";
                }
              }
              std::cout << std::dec << std::endl;
              old_arrival_time64 = arrival_time64;
              // Hand this entry of the ring buffer (frame) back to kernel

              header->tp_status = TP_STATUS_KERNEL;
            } else {
              // this packet is not tp_status_user, poll again

              std::cout << "poll" << std::endl;
              continue;
            }
            // Go to next frame in ring buffer

            idx = ((idx + 1) % rx_buffer_cnt);
            header = reinterpret_cast<tpacket2_hdr *>(base + idx * frame_size);
          } while (header->tp_status & TP_STATUS_USER);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
      }
    }
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  return 0;
}
