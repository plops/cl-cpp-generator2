// no preamble

#include <spdlog/spdlog.h>
extern "C" {
#include <libnc.h>
};
#include "Net.h"
Net::Net() : s{nc_context_init(1)}, dev{nc_new_device(s, "cpu")} {
  spdlog::info("Net constructor");
}
Net::~Net() { nc_context_end(s); }