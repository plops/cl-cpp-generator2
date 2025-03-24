//
// Created by martin on 3/23/25.
//

#ifndef TCPSOCKET_H
#define TCPSOCKET_H
extern "C" {
#include <netinet/in.h>
}
#include "ISocket.h"

class TCPSocket final : public ISocket {
public:
  ~TCPSocket() override;
  bool open(uint16_t port) override;
  void close() override;
  bool send(const std::string &data) override;
  std::string receive() override;

private:
  int sockfd{-1};
  sockaddr_in serverAddress{};
  int padding{0};
};

#endif // TCPSOCKET_H
