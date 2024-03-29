#ifndef UART_H
#define UART_H

#include <fmt/format.h>
#include <vector>
class Uart {
private:
  explicit Uart();

public:
  static Uart &getInstance() {
    static Uart instance;
    return instance;
  }

  // Delete copy constructor and assignment operator

  Uart(Uart const &) = delete;
  Uart &operator=(Uart const &) = delete;
  template <typename... Args>
  void print(fmt::format_string<Args...> fmt, Args &&...args) {
    auto ostr{std::vector<uint8_t>()};
    // Use format_to with a back_inserter to append formatted output to the
    // vector

    fmt::format_to(std::back_inserter(ostr), fmt, std::forward<Args>(args)...);
    SendString(ostr.data(), static_cast<uint16_t>(ostr.size()));
  }

  /** Overload for const char pointer

*/
  void print(const char *str);
  /** Overload for string literals (will not call strlen for known strings)

*/
  template <std::size_t N> void print(const char (&str)[N]) {
    // N includes the null terminator, so we subtract 1 1t oget the actual
    // string length

    SendString(reinterpret_cast<uint8_t *>(const_cast<char *>(str)),
               static_cast<uint16_t>(N - 1));
  }

private:
  void SendString(uint8_t *buf, uint16_t len);
};

#endif /* !UART_H */