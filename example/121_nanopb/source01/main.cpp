#include <deque>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#define FMT_HEADER_ONLY
#include "core.h"

int main(int argc, char **argv) {
  fmt::print("generation date 16:32:48 of Monday, 2023-04-10 (GMT+1)\n");
  auto listenfd = socket(AF_INET, SOCK_STREAM, 0);
  auto reuse = int(1);
  setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  auto servaddr = sockaddr_in();
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;

  servaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  servaddr.sin_port = htons(1234);

  return 0;
}
