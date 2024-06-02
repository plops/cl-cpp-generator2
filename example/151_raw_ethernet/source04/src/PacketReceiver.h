#ifndef PACKETRECEIVER_H
#define PACKETRECEIVER_H

// header
 
#include <cstdint>
#include <string>
#include <vector>
#include <array> 
class PacketReceiver  {
        public:
         PacketReceiver (const std::string& if_name = "lo", const uint32_t& block_size = 4096, const uint32_t& block_nr = 1, const uint32_t& frame_size = 128)       ;   
         ~PacketReceiver ()       ;   
            /** @brief Receives packets from a socket and stores them in a vector.

This function receives packets from a socket and stores them in a vector. 
The vector is passed by reference, allowing the function to modify its contents directly.
At the start of the function, the vector is cleared, removing all elements but not deallocating its memory.
This is done to minimize memory allocations if a similar number of packets is returned each time the function is called.

@param packets A reference to a vector where the received packets will be stored. 
       The vector is cleared at the start of the function, but its memory is not deallocated.
 


    */ 
        void receive (std::vector<std::vector<uint8_t>>& packets)       ;   
        private:
        int sockfd;
        void* mmap_base;
        size_t mmap_size;
        std::string if_name;
        uint32_t block_size;
        uint32_t block_nr;
        uint32_t frame_size;
        uint32_t frame_nr;
        uint32_t rx_buffer_cnt;
};

#endif /* !PACKETRECEIVER_H */