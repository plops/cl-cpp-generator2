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
    SendString(ostr.data(), ostr.size());
  }

  /** Overload for const char pointer

*/
  void print(const char *str);

private:
  void SendString(uint8_t *buf, uint16_t len);
};

#endif /* !UART_H */