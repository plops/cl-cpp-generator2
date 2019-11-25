
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
struct space_packet_header_info_t {
  std::array<unsigned char, 62 + 6> head;
  size_t offset;
};
typedef struct space_packet_header_info_t space_packet_header_info_t;
void destroy_collect_packet_headers() {}
void init_collect_packet_headers() {
  auto data_length = ((((1) * (state._mmap_data[5]))) +
                      (((256) * (((0xFF) & (state._mmap_data[4]))))));
};