/** This C++ code is a low-level network packet sniffer that uses raw
sockets and the Linux packet_mmap API to capture packets from a
network interface.

The main function starts by creating a raw socket and binding it to
the `wlan0` interface. It then sets the socket option to use version 2
of the packet_mmap API (TPACKET_V2).

Next, it configures a ring buffer for packet capture, ensuring that
the block size is a multiple of the system page size and a power of
two, and that it is also a multiple of the frame size.

The ring buffer is then memory-mapped into the process's address
space.

The main loop of the program polls the socket for incoming
packets. When a packet arrives, it iterates through the packets in the
ring buffer, printing out information about each packet, including the
time of arrival, the length of the packet data, and a hex dump of the
first 128 bytes of the packet data.

If a packet is marked as having been copied or lost, it prints out
additional information.

After processing a packet, it hands the frame back to the kernel and
moves on to the next frame in the ring buffer.

If an error occurs during the execution of the program, it catches the
exception and prints out an error message.
*/
#include <arpa/inet.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <linux/if_packet.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <system_error>
#include <thread>
#include <unistd.h>

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
    if (bind(sockfd, reinterpret_cast<sockaddr *>(&ll), sizeof(ll)) < 0) {
      std::cout << "bind error"
                << " errno='" << errno << "' " << std::endl;
    }
    // define version
    auto version{TPACKET_V2};
    setsockopt(sockfd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version));
    // configure ring buffer
    auto block_size{static_cast<uint32_t>(1 * getpagesize())};
    auto block_nr{8U};
    auto frame_size{256U};
    auto frame_nr{(block_size / frame_size) * block_nr};
    auto req{tpacket_req{.tp_block_size = block_size,
                         .tp_block_nr = block_nr,
                         .tp_frame_size = frame_size,
                         .tp_frame_nr = frame_nr}};
    // the following conditions don't have to be strictly fulfilled. the ring
    // buffer
    // in the kernel works with other configurations. but my code to iterate
    // through the blocks of the ring buffer can only handle this
    if (0 != (block_size % getpagesize())) {
      throw std::runtime_error(
          "block_size should be a multiple of getpagesize()");
    }
    if (0 != (block_size % frame_size)) {
      throw std::runtime_error("block_size should be a multiple of frame_size");
    }
    std::cout << ""
              << " block_size='" << block_size << "' "
              << " block_nr='" << block_nr << "' "
              << " frame_size='" << frame_size << "' "
              << " frame_nr='" << frame_nr << "' " << std::endl;
    if (setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) < 0) {
      throw std::runtime_error("setsockopt");
    }
    // map the ring buffer
    auto mmap_size{block_size * block_nr};
    auto mmap_base{mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        sockfd, 0)};
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
              auto arrival_time64{uint64_t(1000000000) * header->tp_sec +
                                  header->tp_nsec};
              auto delta64{arrival_time64 - old_arrival_time64};
              std::cout << std::dec << std::setw(20) << std::setfill(' ')
                        << arrival_time64 << " " << std::setw(12) << delta64
                        << " ";
              for (unsigned int i = 0; i < (data_len < 128U ? data_len : 128U);
                   i += 1) {
                // color sequence bytes of icmp packet in red
                if (27 == i) {
                  std::cout << "\033[31m";
                }
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(data[i]);
                if (27 + 1 == i) {
                  std::cout << "\033[0m";
                }
                if (7 == (i % 8)) {
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
  // unreachable:
  return 0;
}
