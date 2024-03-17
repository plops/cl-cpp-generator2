// no preamble

// implementation

extern "C" {
#include <CH59x_common.h>
};
#include "Uart.h"
/**
- The constructor of Uart is made private to prevent direct
  instantiation.

- A static method getInstance() is provided to get the singleton
  instance.

- Copy constructor and copy assignment operator are deleted to prevent copying.

- Not working on bare metal risc-v yet: A std::mutex named mutex is
  added to protect critical sections within the print method. This
  mutex is locked using std::lock_guard before accessing shared
  resources.

- Please note, using a mutex in a high-frequency logging or in
  interrupt context can lead to performance bottlenecks or deadlocks
  if not handled carefully. Always consider the specific requirements
  and constraints of your embedded system when introducing
  thread-safety mechanisms.


*/
Uart::Uart() {
  // This will configure UART1 to send and receive at 115200 baud:

  /** up to 6Mbps is possible. fifo can store 8 bytes

*/
  GPIOA_SetBits(GPIO_Pin_9);
  GPIOA_ModeCfg(GPIO_Pin_8, GPIO_ModeIN_PU);
  GPIOA_ModeCfg(GPIO_Pin_9, GPIO_ModeOut_PP_5mA);
  UART1_DefInit();
}
void Uart::print(const char *str) {
  auto n{strlen(str)};
  SendString(reinterpret_cast<uint8_t *>(const_cast<char *>(str)),
             static_cast<uint16_t>(n));
}
void Uart::SendString(uint8_t *buf, uint16_t len) {
  // FIXME: hold mutex here

  UART1_SendString(buf, len);
}