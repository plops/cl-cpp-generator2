#ifndef PACKETRECEIVER_H
#define PACKETRECEIVER_H

// header
 
#include <cstdint> 
class PacketReceiver  {
        public:
         PacketReceiver (const std::string& if_name = "lo", const uint32& block_size = 4096, const uint32& block_nr = 1, const uint32& frame_size = 128)       ;   
         ~PacketReceiver ()       ;   
        const int& GetSockfd () const      ;   
        void SetSockfd (int sockfd)       ;   
        void* GetMmapBase ()       ;   
        void SetMmapBase (void* mmap_base)       ;   
        const size_t& GetMmapSize () const      ;   
        void SetMmapSize (size_t mmap_size)       ;   
        const std::string& GetIfName () const      ;   
        void SetIfName (std::string if_name)       ;   
        const uint32& GetBlockSize () const      ;   
        void SetBlockSize (uint32 block_size)       ;   
        const uint32& GetBlockNr () const      ;   
        void SetBlockNr (uint32 block_nr)       ;   
        const uint32& GetFrameSize () const      ;   
        void SetFrameSize (uint32 frame_size)       ;   
        const uint32& GetFrameNr () const      ;   
        void SetFrameNr (uint32 frame_nr)       ;   
        const uint32& GetRxBufferCnt () const      ;   
        void SetRxBufferCnt (uint32 rx_buffer_cnt)       ;   
        private:
        int sockfd;
        void* mmap_base;
        size_t mmap_size;
        std::string if_name;
        uint32 block_size;
        uint32 block_nr;
        uint32 frame_size;
        uint32 frame_nr;
        uint32 rx_buffer_cnt;
};

#endif /* !PACKETRECEIVER_H */