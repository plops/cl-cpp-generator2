//
// Created by martin on 3/23/25.
//

#ifndef TCPSOCKET_H
#define TCPSOCKET_H
#include "ISocket.h"
#include <cstdint>
// extern "C" {
#include <netinet/in.h>
// }

class TCPSocket : public ISocket {
public:
  ~TCPSocket() override;
  bool open(int port) override;
  void close() override;
  bool send(const std::string &data) override;
  std::string receive() override;

private:
  int sockfd;
  sockaddr_in serverAddress;
};

#endif // TCPSOCKET_H
