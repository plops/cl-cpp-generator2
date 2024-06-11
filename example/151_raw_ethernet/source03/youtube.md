## Video Outline and Transcript:

**Intro**

```cpp
// No code shown in this segment
```

Hello everyone, and welcome back to the channel! Today, we're diving deep into the world of low-level networking with a fascinating project: building a network packet sniffer from scratch using C++. 

Imagine being able to peek under the hood of your network traffic, seeing the raw data flowing between your computer and the internet. That's exactly what a packet sniffer allows you to do, and we'll explore how to build one step-by-step. 

Get ready to get your hands dirty with raw sockets, ring buffers, and the magic of memory mapping. Let's get started!

**Understanding Network Packets**

```cpp
// No code shown in this segment
```

Before we jump into the code, let's take a moment to understand what network packets are and why they're crucial for network communication. 

In essence, network packets are like tiny envelopes carrying pieces of information across networks. Just like a physical envelope has an address, contents, and other markings, network packets contain source and destination addresses, data, and control information. 

Our packet sniffer will capture these packets in transit, giving us a glimpse into the raw data exchange happening on our network. 

**Creating a Raw Socket**

```cpp
#include <arpa/inet.h>
#include <cstdint>
// ... other includes

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
    // ... rest of the code
  } catch (const std::system_error &ex) {
    std::cerr << "Error: " << ex.what() << " (" << ex.code() << ")\n";
    return 1;
  }
  // unreachable:
  return 0;
}
```

Now, let's dive into the code. The first step is to create a raw socket.  Unlike regular sockets, which operate at a higher level, raw sockets give us direct access to the network interface, allowing us to capture all packets passing through. 

We use the `socket()` function with the `AF_PACKET` address family, signifying that we're working with network packets.  The `SOCK_RAW` socket type indicates a raw socket, and `htons(ETH_P_ALL)` captures all Ethernet packet types. 

We also check if the socket creation was successful. If not, it might be a permissions issue, so we'll display an error message.

**Binding the Socket and Setting Options**

```cpp
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
```

With our raw socket created, we need to bind it to a specific network interface. This tells the system which interface's traffic we want to capture. 

We use the `if_nametoindex()` function to obtain the interface index of "wlan0" and then bind our socket to this interface using the `bind()` function.

Next, we set a crucial socket option: the packet_mmap API version. Version 2 (`TPACKET_V2`) provides a more efficient way to capture packets using ring buffers, which we'll explore shortly.

**Configuring the Ring Buffer**

```cpp
    // configure ring buffer
    auto block_size{static_cast<uint32_t>(1 * getpagesize())};
    auto block_nr{8U};
    auto frame_size{256U};
    auto frame_nr{(block_size / frame_size) * block_nr};
    auto req{tpacket_req{.tp_block_size = block_size,
                         .tp_block_nr = block_nr,
                         .tp_frame_size = frame_size,
                         .tp_frame_nr = frame_nr}};
    // ... checks for block size and frame size
    std::cout << ""
              << " block_size='" << block_size << "' "
              << " block_nr='" << block_nr << "' "
              << " frame_size='" << frame_size << "' "
              << " frame_nr='" << frame_nr << "' " << std::endl;
    if (setsockopt(sockfd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) < 0) {
      throw std::runtime_error("setsockopt");
    }
```

Now comes the clever part: setting up a ring buffer for efficient packet capture. Instead of copying packets from kernel space to user space individually, a ring buffer acts as a shared memory region where the kernel writes incoming packets. 

We define the `block_size`, `block_nr`, `frame_size`, and `frame_nr` to configure the ring buffer's structure. We ensure that the block size is a multiple of the system page size and a power of two for optimal performance.

Then, using `setsockopt()`, we apply these settings to our socket, activating the ring buffer mechanism.

**Memory Mapping and Packet Processing**

```cpp
    // map the ring buffer
    auto mmap_size{block_size * block_nr};
    auto mmap_base{mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        sockfd, 0)};
    // ... other variables
    while (true) {
      // ... polling and packet processing code
    }
```

With the ring buffer ready, we map it into our process's address space using `mmap()`. This allows us to access the ring buffer's contents directly as if it were regular memory, further boosting efficiency.

Now, our program enters the main loop. It continuously polls the socket for incoming packets. When a packet arrives, we iterate through the ring buffer, extracting information from each captured packet. 

We'll delve deeper into the packet processing details in the next segment.

**Extracting Packet Information**

```cpp
          // Iterate through packets in the ring buffer
          do {
            if (header->tp_status & TP_STATUS_USER) {
              // ... process packet data
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
```

For each packet, we extract relevant information like the arrival time, data length, and a hexadecimal dump of the first 128 bytes of the packet data. 

We also check for any packet loss or copying indicators, providing valuable insights into the network's health and performance.

Finally, after processing a packet, we mark the corresponding frame in the ring buffer as available for the kernel to use again. This seamless handover ensures continuous packet capture without interruptions.



## Video Outline and Transcript: (Updated with Packet Loss Section)

**Intro**

```cpp
// No code shown in this segment
```

Hello everyone, and welcome back to the channel! Today, we're diving deep into the world of low-level networking with a fascinating project: building a network packet sniffer from scratch using C++. 

Imagine being able to peek under the hood of your network traffic, seeing the raw data flowing between your computer and the internet. That's exactly what a packet sniffer allows you to do, and we'll explore how to build one step-by-step. 

Get ready to get your hands dirty with raw sockets, ring buffers, and the magic of memory mapping. We'll also discuss how even with these powerful techniques, packet loss can occur and how to gather insights into these losses. Let's get started!

**Understanding Packet Loss: Even with Packet MMAP**

```cpp
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
              // ... other packet processing code
            } 
            // ... other code
          } while (header->tp_status & TP_STATUS_USER);
```

You might wonder, why does packet loss happen even with the efficient packet mmap interface?  

Even though packet mmap reduces overhead, our sniffer still operates in user space. If the network traffic is exceptionally high, and our sniffer can't process packets quickly enough to free up space in the ring buffer, the kernel might be forced to drop incoming packets.

Fortunately, the packet mmap API provides mechanisms to detect and quantify this loss. The `TP_STATUS_LOSING` flag within the packet header signals potential packet loss.

Furthermore, we can use the `PACKET_STATISTICS` socket option to retrieve statistics about the ring buffer's performance. This information, accessible through the `tpacket_stats` structure, reveals valuable metrics like the total number of dropped packets (`tp_drops`) and the total number of packets received (`tp_packets`).

By monitoring these statistics, we can gain insights into the extent of packet loss and potentially adjust our sniffer's configuration or processing logic to minimize it.


**Conclusion**

```cpp
// No code shown in this segment
```

And there you have it! We've successfully built a powerful network packet sniffer capable of capturing and analyzing raw network traffic.

This project gave us a glimpse into the exciting world of low-level networking, raw sockets, ring buffers, and memory mapping.

Feel free to experiment with the code, analyze different network interfaces, and even extend the sniffer's capabilities.

Don't forget to like, subscribe, and share this video with your fellow coding enthusiasts. Until next time, happy coding!
