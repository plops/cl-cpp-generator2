extern "C" {
#include <CH59x_common.h>
#include <CH59x_pwr.h>
#include <CH59x_sys.h>
#include <board.h>
};
#include <cstdio>

int main() {
  // Enable DCDC

  PWR_DCDCCfg(ENABLE);
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  board_button_init();
  board_led_init();
  // the blue led flashes. the BOOT button switches between a fast and a slow
  // flashing frequency

  auto tick{uint32_t(0)};
  auto toggle_tick{uint32_t(250)};
  while (1) {
    tick++;
    if (0 == (tick % toggle_tick)) {
      board_led_toggle();
    }
    if (board_button_getstate()) {
      while (board_button_getstate()) {
        DelayMs(50);
      }
      if (250 == toggle_tick) {
        toggle_tick = 100;
      } else {
        toggle_tick = 250;
      }
    }
    DelayMs(1);
  }
  return 0;
}
