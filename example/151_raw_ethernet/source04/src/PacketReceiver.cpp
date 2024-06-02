// no preamble

// implementation

#include "PacketReceiver.h"
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
PacketReceiver::PacketReceiver(
    std::function<void(const uint8_t *, const size_t)> callback,
    const std::string &if_name, const uint32_t &block_size,
    const uint32_t &block_nr, const uint32_t &frame_size)
    : callback{std::move(callback)}, sockfd{-1}, mmap_base{nullptr},
      mmap_size{0}, if_name{if_name}, block_size{block_size},
      block_nr{block_nr}, frame_size{frame_size} {
  auto sockfd{socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))};
  if (sockfd < 0) {
    throw std::runtime_error("error opening socket. try running as root");
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
    throw std::runtime_error("bind error");
  }
  // define version

  auto version{TPACKET_V2};
  setsockopt(sockfd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version));
  // configure ring buffer

  frame_nr = (block_size / frame_size) * block_nr;
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

  mmap_size = block_size * block_nr;
  mmap_base =
      mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, sockfd, 0);
  auto rx_buffer_size{block_size * block_nr};
  auto rx_buffer_addr{mmap_base};
  // rx_buffer_cnt is the number of frames in the ring buffer

  rx_buffer_cnt = ((block_size * block_nr) / frame_size);
  std::cout << ""
            << " rx_buffer_size='" << rx_buffer_size << "' "
            << " rx_buffer_cnt='" << rx_buffer_cnt << "' " << std::endl;
}
PacketReceiver::~PacketReceiver() {
  // Disable PACKET_RX_RING

  auto req{tpacket_req{.tp_block_size = 0,
                       .tp_block_nr = 0,
                       .tp_frame_size = 0,
                       .tp_frame_nr = 0}};
  setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));
  // Unmap the memory-mapped buffer

  munmap(mmap_base, mmap_size);
  // Close the socket

  close(sockfd);
}
void PacketReceiver::receive() {
  auto idx{0U};
  auto old_arrival_time64{uint64_t(0)};
  while (true) {
    auto pollfds{pollfd({.fd = sockfd, .events = POLLIN, .revents = 0})};
    auto poll_res{ppoll(&pollfds, 1, nullptr, nullptr)};
    if (poll_res < 0) {
      std::cout << "error in ppoll"
                << " poll_res='" << poll_res << "' "
                << " errno='" << errno << "' " << std::endl;
      throw std::runtime_error("ppoll error");
    } else if (poll_res == 0) {
      std::cout << "timeout" << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    } else {
      if (POLLIN & pollfds.revents) {
        auto base{reinterpret_cast<uint8_t *>(mmap_base)};
        auto header{reinterpret_cast<tpacket2_hdr *>(base + idx * frame_size)};
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
            auto arrival_time64{1000000000 * header->tp_sec + header->tp_nsec};
            auto delta64{arrival_time64 - old_arrival_time64};
            auto delta_ms{static_cast<double>(delta64) / 1.00e+6};
            auto arrival_timepoint{
                std::chrono::system_clock::from_time_t(header->tp_sec) +
                std::chrono::nanoseconds(header->tp_nsec)};
            auto time{std::chrono::system_clock::to_time_t(arrival_timepoint)};
            auto local_time{std::localtime(&time)};
            auto local_time_hr{std::put_time(local_time, "%Y-%m-%d %H:%M:%S")};
            std::cout << local_time_hr << "." << std::dec << std::setw(6)
                      << (header->tp_nsec / 1000) << " " << std::setfill(' ')
                      << std::setw(5 + 6) << std::fixed << std::setprecision(6)
                      << (delta_ms < 10000 ? std::to_string(delta_ms)
                                           : "xxxx.xxxxxx")
                      << " " << std::dec << std::setw(6) << data_len << " "
                      << std::setw(4) << idx << " ";
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
            callback(data, data_len);
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
}
const std::function<void(const uint8_t *, size_t)> &
PacketReceiver::GetCallback() const {
  return callback;
}
void PacketReceiver::SetCallback(
    std::function<void(const uint8_t *, size_t)> callback) {
  this->callback = callback;
}
const int &PacketReceiver::GetSockfd() const { return sockfd; }
void PacketReceiver::SetSockfd(int sockfd) { this->sockfd = sockfd; }
void *PacketReceiver::GetMmapBase() { return mmap_base; }
void PacketReceiver::SetMmapBase(void *mmap_base) {
  this->mmap_base = mmap_base;
}
const size_t &PacketReceiver::GetMmapSize() const { return mmap_size; }
void PacketReceiver::SetMmapSize(size_t mmap_size) {
  this->mmap_size = mmap_size;
}
const std::string &PacketReceiver::GetIfName() const { return if_name; }
void PacketReceiver::SetIfName(std::string if_name) { this->if_name = if_name; }
const uint32_t &PacketReceiver::GetBlockSize() const { return block_size; }
void PacketReceiver::SetBlockSize(uint32_t block_size) {
  this->block_size = block_size;
}
const uint32_t &PacketReceiver::GetBlockNr() const { return block_nr; }
void PacketReceiver::SetBlockNr(uint32_t block_nr) {
  this->block_nr = block_nr;
}
const uint32_t &PacketReceiver::GetFrameSize() const { return frame_size; }
void PacketReceiver::SetFrameSize(uint32_t frame_size) {
  this->frame_size = frame_size;
}
const uint32_t &PacketReceiver::GetFrameNr() const { return frame_nr; }
void PacketReceiver::SetFrameNr(uint32_t frame_nr) {
  this->frame_nr = frame_nr;
}
const uint32_t &PacketReceiver::GetRxBufferCnt() const { return rx_buffer_cnt; }
void PacketReceiver::SetRxBufferCnt(uint32_t rx_buffer_cnt) {
  this->rx_buffer_cnt = rx_buffer_cnt;
}