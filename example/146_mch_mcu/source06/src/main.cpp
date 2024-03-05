// based on https://github.com/openwch/ch592/blob/main/EVT/EXAM/UART1/src/Main.c

// write debug message on UART

#include <array>
#include <cassert>
extern "C" {
#include <CH59x_common.h>
};
auto TxBuf{std::array<uint8_t, 20>("THis s a tx test\r\n")};
auto RxBuf{std::array<uint8_t, 100>()};
auto trigB{uint8_t(0)};

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  auto len{uint8_t(0)};
  GPIOA_SetBits(GPIO_Pin_9);
  GPIOA_ModeCfg(GPIO_Pin_8, GPIO_ModeIN_PU);
  GPIOA_ModeCfg(GPIO_Pin_9, GPIO_ModeOut_PP_5mA);
  UART1_DefInit();
}
