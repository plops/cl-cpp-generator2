#include <iostream>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <tins/tins.h>

int main(int argc, char **argv) {
  spdlog::info("start  argc='{}'", argc);
  auto op{popl::OptionParser("allowed opitons")};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  op.parse(argc, argv);
  if (helpOption->count()) {
    (std::cout) << (op) << (std::endl);
    exit(0);
  }
  auto vt{std::vector<Tins::Packet>()};
  auto sniffer{Tins::Sniffer("wlp4s0")};
  while ((vt.size()) < (10)) {
    vt.push_back(sniffer.next_packet());
  }
  for (const auto &&packet : vt) {
    if ((packet.pdu())->(find_pdu<Tins::IP>())) {
      spdlog::info("  packet.timestamp().seconds()='{}'  "
                   "(packet.pdu())->(rfind_pdu<Tins::IP>().src_addr())='{}'",
                   packet.timestamp().seconds(),
                   (packet.pdu())->(rfind_pdu<Tins::IP>().src_addr()));
    }
  }
}
