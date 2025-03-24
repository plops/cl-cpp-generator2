//
// Created by martin on 3/24/25.
//

#ifndef UDPSOCKET_H
#define UDPSOCKET_H
extern "C" {
#include <netinet/in.h>
}
#include "ISocket.h"

class UDPSocket : public ISocket {
public:
  ~UDPSocket() override;
  bool open(int port) override;
  void close() override;
  bool send(const std::string &data) override;
  std::string receive() override;

private:
  int sockfd;
  sockaddr_in serverAddress;
  sockaddr_in clientAddress;
  socklen_t clientAddressLength;
};

#endif // UDPSOCKET_H
