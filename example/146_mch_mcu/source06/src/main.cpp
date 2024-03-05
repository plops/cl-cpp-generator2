// based on https://github.com/openwch/ch592/blob/main/EVT/EXAM/UART1/src/Main.c

// write debug message on UART

#include <array>
#include <cassert>
extern "C" {
#include <CH59x_common.h>
};

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  GPIOA_SetBits(GPIO_Pin_9);
  GPIOA_ModeCfg(GPIO_Pin_8, GPIO_ModeIN_PU);
  GPIOA_ModeCfg(GPIO_Pin_9, GPIO_ModeOut_PP_5mA);
  // This will configure UART to send and receive at 115200 baud:

  UART1_DefInit();
  auto TxBuf{std::array<uint8_t, 20>("This s a tx test\r\n")};
  if (1) {
    UART1_SendString(TxBuf.data(), TxBuf.size());
  }
  if (1) {
    auto RxBuf{std::array<uint8_t, 100>()};
    while (true) {
      auto len{UART1_RecvString(RxBuf.data())};
      if (len) {
        UART1_SendString(RxBuf.data(), len);
      }
    }
  }
}
