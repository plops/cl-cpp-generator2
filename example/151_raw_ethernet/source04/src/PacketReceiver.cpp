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
PacketReceiver::PacketReceiver(const std::string &if_name,
                               const uint32_t &block_size,
                               const uint32_t &block_nr,
                               const uint32_t &frame_size,
                               const PacketReceivedCallback &callback)
    : sockfd{-1}, mmap_base{nullptr}, mmap_size{0}, if_name{if_name},
      block_size{block_size}, block_nr{block_nr}, frame_size{frame_size},
      frame_nr{0}, rx_buffer_cnt{0}, callback{callback} {
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
      throw std::runtime_error("ppoll error");
    } else if (poll_res == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    } else {
      if (POLLIN & pollfds.revents) {
        auto base{reinterpret_cast<uint8_t *>(mmap_base)};
        auto header{reinterpret_cast<tpacket2_hdr *>(base + idx * frame_size)};
        // Iterate through packets in the ring buffer

        do {
          if (header->tp_status & TP_STATUS_USER) {

            auto data{reinterpret_cast<uint8_t *>(header) + header->tp_net};
            auto data_len{header->tp_snaplen};
            callback(data, data_len);
            // Hand this entry of the ring buffer (frame) back to kernel

            header->tp_status = TP_STATUS_KERNEL;
          } else {
            // this packet is not tp_status_user, poll again

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
const PacketReceivedCallback &PacketReceiver::GetCallback() const {
  return callback;
}
void PacketReceiver::SetCallback(PacketReceivedCallback callback) {
  this->callback = callback;
}