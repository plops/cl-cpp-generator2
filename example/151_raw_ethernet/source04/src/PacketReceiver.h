#ifndef PACKETRECEIVER_H
#define PACKETRECEIVER_H

// header
 
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <functional> 
class PacketReceiver  {
        public:
         PacketReceiver (std::function<void(const uint8_t*, const size_t)> callback = nullptr, const std::string& if_name = "lo", const uint32_t& frame_size = 128, const uint32_t& block_size = 4096, const uint32_t& block_nr = 1)       ;   
         ~PacketReceiver ()       ;   
            /** 
@brief Continuously reads packets from a socket and processes them.

This function runs an infinite loop that continuously polls a socket for incoming packets.
When a packet is received, it is processed and the provided callback function is called with the packet data.
The function also handles various status flags and errors, and provides detailed logging of packet information and status.

The function uses a ring buffer mechanism for efficient packet reading. It keeps track of the current index in the ring buffer,
and after processing each packet, it hands the buffer back to the kernel and moves to the next buffer in the ring.

If the function encounters an error during polling, it throws a runtime_error exception.

If no packets are available for reading, the function sleeps for 4 milliseconds before polling the socket again.

@note This function runs indefinitely until an error occurs or the program is terminated.
 


    */ 
        void receive ()       ;   
        
        const std::function<void(const uint8_t*, size_t)>& GetCallback () const      ;   
        void SetCallback (std::function<void(const uint8_t*, size_t)> callback)       ;   
        
        const int& GetSockfd () const      ;   
        void SetSockfd (int sockfd)       ;   
        
        void* GetMmapBase ()       ;   
        void SetMmapBase (void* mmap_base)       ;   
        
        const size_t& GetMmapSize () const      ;   
        void SetMmapSize (size_t mmap_size)       ;   
        
        const std::string& GetIfName () const      ;   
        void SetIfName (std::string if_name)       ;   
        
        const uint32_t& GetFrameSize () const      ;   
        void SetFrameSize (uint32_t frame_size)       ;   
        
        const uint32_t& GetBlockSize () const      ;   
        void SetBlockSize (uint32_t block_size)       ;   
        
        const uint32_t& GetBlockNr () const      ;   
        void SetBlockNr (uint32_t block_nr)       ;   
        
        const uint32_t& GetFrameNr () const      ;   
        void SetFrameNr (uint32_t frame_nr)       ;   
            /** The number of frames in the RX ring buffer.

    */ 
        const uint32_t& GetRxBufferCnt () const      ;   
        void SetRxBufferCnt (uint32_t rx_buffer_cnt)       ;   
        private:
        std::function<void(const uint8_t*, size_t)> callback;
        int sockfd;
        void* mmap_base;
        size_t mmap_size;
        std::string if_name;
        uint32_t frame_size;
        uint32_t block_size;
        uint32_t block_nr;
        uint32_t frame_nr;
        uint32_t rx_buffer_cnt;
};

#endif /* !PACKETRECEIVER_H */