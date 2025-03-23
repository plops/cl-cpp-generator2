//
// Created by martin on 3/23/25.
//

#include "TCPSocket.h"
#include <array>
#include <string>

extern "C" {
#include <sys/socket.h>
#include <unistd.h>
}

using namespace std;

TCPSocket::~TCPSocket() override {
  if (-1 != sockfd) {
    ::close(sockfd);
    sockfd = -1;
  }
}
bool TCPSocket::open(int port) {
  sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (-1 == sockfd) {
    perror("socket creation failed");
    return false;
  }
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY; // Bind to all interfaces
  serverAddress.sin_port = htons(port);

  if (-1 == ::bind(sockfd, reinterpret_cast<struct sockaddr *>(&serverAddress),
                   sizeof(serverAddress))) {
    perror("bind failed");
    ::close(sockfd);
    sockfd = -1;
    return false;
  }

  if (-1 == ::listen(sockfd, SOMAXCONN)) { // Listen with a backlog of 4096
    perror("listen failed");
    ::close(sockfd);
    sockfd = -1;
    return false;
  }
  return true;
}
void TCPSocket::close() {
  if (-1 != sockfd) {
    ::close(sockfd);
    sockfd = -1;
  }
}
bool TCPSocket::send(const std::string &data) {
  int clientSockfd = ::accept(sockfd, nullptr, nullptr);
  if (-1 == clientSockfd) {
    perror("accept failed");
    return false;
  }
  auto bytesSent = ::send(clientSockfd, data.c_str(), data.size(), 0);
  ::close(clientSockfd); // close connection after sending
  if (-1 == bytesSent) {
    perror("send failed");
    return false;
  }
  return true;
}
std::string TCPSocket::receive() {
  auto clientSockfd = ::accept(sockfd, nullptr, nullptr);
  if (-1 == clientSockfd) {
    perror("accept failed");
    return "";
  }
  array<char, 1024> buffer{0};
  auto bytesReceived =
      ::recv(clientSockfd, buffer.data(), buffer.size() - 1, 0);
  ::close(clientSockfd);
  if (-1 == bytesReceived) {
    perror("recv failed");
    return "";
  }
  buffer[bytesReceived] = '\0'; // Null-terminate the received data
  return string{buffer.begin(), buffer.begin() + bytesReceived};
}