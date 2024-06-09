```cpp
#include <iostream>
#include <vector>
#include <cstdint>

// Simplified representation of network headers
struct EthernetHeader {
  uint8_t destMac[6];
  uint8_t srcMac[6];
  uint16_t etherType;
};

struct IPv4Header {
  uint8_t versionAndIHL;
  uint8_t dscpAndECN;
  uint16_t totalLength;
  uint16_t identification;
  uint16_t flagsAndFragmentOffset;
  uint8_t timeToLive;
  uint8_t protocol;
  uint16_t headerChecksum;
  uint8_t sourceIp[4];
  uint8_t destIp[4];
};

struct ICMPHeader {
  uint8_t type;
  uint8_t code;
  uint16_t checksum;
  uint16_t identifier;
  uint16_t sequenceNumber; 
  // ... other ICMP fields
};

// Function to parse the sequence number from a raw packet
uint16_t getIcmpSequenceNumber(const std::vector<uint8_t>& packet) {
  // Check for minimum packet size
  if (packet.size() < sizeof(EthernetHeader) + sizeof(IPv4Header) + sizeof(ICMPHeader)) {
    throw std::runtime_error("Packet too small!");
  }

  // Cast raw data to header structures for easier access
  const EthernetHeader* eth = reinterpret_cast<const EthernetHeader*>(packet.data());
  const IPv4Header* ip = reinterpret_cast<const IPv4Header*>(packet.data() + sizeof(EthernetHeader));
  const ICMPHeader* icmp = reinterpret_cast<const ICMPHeader*>(packet.data() + sizeof(EthernetHeader) + sizeof(IPv4Header));

  // Basic sanity checks (you can add more)
  if (eth->etherType != htons(0x0800)) { // IPv4 ethertype
    throw std::runtime_error("Not an IPv4 packet!");
  }
  if (ip->protocol != 1) { // ICMP protocol number
    throw std::runtime_error("Not an ICMP packet!");
  }

  return ntohs(icmp->sequenceNumber);
}

int main() {
  // Example usage
  std::vector<uint8_t> examplePacket = { 
    // ... (Fill with your sample packet data)
  };

  try {
    uint16_t sequenceNum = getIcmpSequenceNumber(examplePacket);
    std::cout << "ICMP Sequence Number: " << sequenceNum << std::endl; 
  } catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}
```

**Explanation:**

1. **Simplified Headers:** The code defines structs for `EthernetHeader`, `IPv4Header`, and `ICMPHeader`, only including necessary fields for this task.
2. **`getIcmpSequenceNumber` Function:**
   - Takes a `std::vector<uint8_t>` representing the raw packet data.
   - Performs basic size and protocol checks to ensure it's an IPv4 + ICMP packet.
   - Uses `reinterpret_cast` to access the appropriate header structs within the raw data.
   - **Important:** Uses `ntohs` to convert the `sequenceNumber` field from network byte order to host byte order.
3. **Error Handling:** Uses a `try-catch` block to handle potential errors like invalid packet size or protocol mismatches.
4. **Example Usage:** The `main` function demonstrates how to use the `getIcmpSequenceNumber` function with sample packet data.

**Remember:**

- This is a simplified example. Real-world packet processing often involves more complex parsing, checksum validation, and handling of various ICMP message types.
- You need to fill in `examplePacket` with actual packet data (e.g., captured from a network interface) for this code to work.

This improved example provides a more realistic and error-resistant way to parse the ICMP 
