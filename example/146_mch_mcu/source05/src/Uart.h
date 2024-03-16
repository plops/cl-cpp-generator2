#ifndef UART_H
#define UART_H

#include <cstdint>
#include <format.h>
#include <vector>
class Uart {
public:
  explicit Uart();
  template <typename... Args>
  void print(fmt::format_string<Args...> fmt, Args &&...args) {
    auto ostr{std::vector<uint8_t>()};
    // Use format_to with a back_inserter to append formatted output to the
    // vector

    fmt::format_to(std::back_inserter(ostr), fmt, std::forward<Args>(args)...);
    SendString(ostr.data(), ostr.size());
  }

private:
  void SendString(uint8_t *buf, uint16_t len);
};

#endif /* !UART_H */