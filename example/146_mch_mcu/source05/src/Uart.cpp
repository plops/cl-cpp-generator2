// no preamble

// implementation

extern "C" {
#include <CH59x_common.h>
};
#include "Uart.h"
Uart::Uart() {
  // This will configure UART1 to send and receive at 115200 baud:

  /** up to 6Mbps is possible. fifo can store 8 bytes

*/
  GPIOA_SetBits(GPIO_Pin_9);
  GPIOA_ModeCfg(GPIO_Pin_8, GPIO_ModeIN_PU);
  GPIOA_ModeCfg(GPIO_Pin_9, GPIO_ModeOut_PP_5mA);
  UART1_DefInit();
}
void Uart::SendString(uint8_t *buf, uint16_t len) {
  UART1_SendString(buf, len);
}