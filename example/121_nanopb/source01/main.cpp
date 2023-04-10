#include <deque>
extern "C" {
#include "data.pb.h"
#include <netinet/in.h>
#include <pb.h>
#include <pb_decode.h>
#include <pb_encode.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
};
#define FMT_HEADER_ONLY
#include "core.h"

pb_istream_t pb_istream_from_socket(int fd) {
  auto stream = pb_istream_t();
  return stream;
}

void handle_connection(int connfd) {
  auto input = pb_istream_from_socket(connfd);
  auto request = DataRequest();
  if (!(pb_decode_delimited(&input, DataRequest_fields, &request))) {
  }
}

int main(int argc, char **argv) {
  fmt::print("generation date 20:32:23 of Monday, 2023-04-10 (GMT+1)\n");
  auto listenfd = socket(AF_INET, SOCK_STREAM, 0);
  auto reuse = int(1);
  setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  auto servaddr = sockaddr_in();
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;

  servaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  servaddr.sin_port = htons(1234);

  if (!(0 == bind(listenfd, reinterpret_cast<sockaddr *>(&servaddr),
                  sizeof(servaddr)))) {
    fmt::print("error bind\n");
  }
  if (!(0 == listen(listenfd, 5))) {
    fmt::print("error listen\n");
  }
  while (true) {
    auto connfd = accept(listenfd, nullptr, nullptr);
    if (connfd < 0) {
      fmt::print("error accept\n");
    }
    handle_connection(connfd);
    close(connfd);
  }

  return 0;
}
