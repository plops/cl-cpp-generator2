#include "/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/inc/CH59x_common.h"
#include "/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/APP/include/broadcaster.h"
#include "/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/HAL/include/HAL.h"

int main() {
  // Enable DCDC

  PWR_DCDCCfg(ENABLE);
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  // Enable Sleep.

  GPIOA_ModeCfg(GPIO_Pin_All, GPIO_ModeIN_PU);
  GPIOB_ModeCfg(GPIO_Pin_All, GPIO_ModeIN_PU);
  // For Debugging

  GPIOA_SetBits(bTXD1);
  GPIOA_ModeCfg(bTXD1, GPIO_ModeOut_PP_5mA);
  UART1_DefInit();
  PRINT("%s\n", VER_LIB);
  return 0;
}
