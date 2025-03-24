//
// Created by martin on 3/24/25.
//

#include "UDPSocket.h"

#include <array>
#include <iostream>
#include <unistd.h>

using namespace std;

UDPSocket::~UDPSocket() {
  if (sockfd != -1) {
    ::close(sockfd);
    sockfd = -1;
  }
}
bool UDPSocket::open(uint16_t port) {
  sockfd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd == -1) {
    perror("UDPSocket creation failed");
    return false;
  }
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY;
  serverAddress.sin_port = htons(port);
  if (::bind(sockfd, reinterpret_cast<const struct sockaddr *>(&serverAddress),
             sizeof(serverAddress)) == -1) {
    perror("UDPSocket bind failed");
    ::close(sockfd);
    return false;
  }
  return true;
}
void UDPSocket::close() {
  if (sockfd != -1) {
    ::close(sockfd);
    sockfd = -1;
  }
  cout << "UDPSocket closed" << endl;
}
bool UDPSocket::send(const std::string &data) {
  if (0 == clientAddressLength) {
    perror("client address length is 0");
    return false;
  }
  auto bytesSent =
      ::sendto(sockfd, data.data(), data.size(), 0,
               reinterpret_cast<const struct sockaddr *>(&clientAddress),
               clientAddressLength);
  if (bytesSent == -1) {
    perror("UDPSocket sendto");
    return false;
  }
  return true;
}
std::string UDPSocket::receive() {
  array<char, 1024> buffer{0};
  clientAddressLength = sizeof(clientAddress); // Reest length each time
  auto bytesReceived =
      ::recvfrom(sockfd, buffer.data(), buffer.size() - 1, 0,
                 reinterpret_cast<struct sockaddr *>(&clientAddress),
                 &clientAddressLength);
  if (bytesReceived == -1) {
    perror("UDPSocket receive failed");
    return "";
  }
  buffer[bytesReceived] = '\0';
  return string(buffer.begin(), buffer.begin() + bytesReceived);
}
