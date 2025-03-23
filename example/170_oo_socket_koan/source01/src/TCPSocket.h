//
// Created by martin on 3/23/25.
//

#ifndef TCPSOCKET_H
#define TCPSOCKET_H

class TCPSocket : public ISocket {
public:
  ~TCPSocket() override {
    if (-1 != sockfd) {
      ::close(sockfd);
      sockfd = -1;
    }
  }

private:
  int sockfd;
  sockaddr_in address;
};

#endif // TCPSOCKET_H
