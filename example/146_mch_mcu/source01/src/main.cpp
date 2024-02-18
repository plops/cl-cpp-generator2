extern "C" {
#include <CH59x_common.h>
#include <HAL.h>
#include <broadcaster.h>
};
#include <cstdio>
__HIGH_CODE __attribute__((noinline)) void Main_Circulation() {
  while (1) {
    TMOS_SystemProcess();
  }
}

void key_callback(uint8_t keys) {
  if ((keys & HAL_KEY_SW_1)) {
    printf("key pressed\n");
    HalLedSet(HAL_LED_ALL, HAL_LED_MODE_OFF);
    HalLedBlink(1, 2, 30, 1000);
  }
}

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  PRINT("%s\n", VER_LIB);
  CH59x_BLEInit();
  HalKeyConfig(key_callback);
  GAPRole_BroadcasterInit();
  Broadcaster_Init();
  Main_Circulation();
  return 0;
}
