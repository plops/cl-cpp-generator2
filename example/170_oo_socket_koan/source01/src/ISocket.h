//
// Created by martin on 3/23/25.
//

#ifndef ISOCKET_H
#define ISOCKET_H
#include <string>
#include <cstdint>

class ISocket {
public:
  virtual ~ISocket() = default;
  virtual bool open(uint16_t port) = 0;
  virtual void close() = 0;
  virtual bool send(const std::string &data) = 0;
  virtual std::string receive() = 0;
};

#endif // ISOCKET_H
