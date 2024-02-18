// based on https://github.com/openwch/ch592/tree/main/EVT/EXAM/BLE/RF_PHY/APP

// try rf communication module in basic mode

extern "C" {
#include <CH59x_common.h>
#include <CH59x_pwr.h>
#include <CH59x_sys.h>
#include <CONFIG.h>
#include <HAL.h>
#include <board.h>
};
#include <cstdio>

void RF_2G4StatusCallback(uint8_t sta, uint8_t crc, uint8_t *rxBuf) {
  switch (sta) {
  case TX_MODE_TX_FINISH: {
    break;
  };
  case TX_MODE_TX_FAIL: {
    break;
  };
  }
}

uint16_t RF_ProcessEvent(uint8_t task_id, uint16_t events) {
  if ((events & SYS_EVENT_MSG)) {
  }
}

void RF_Init() {
  auto cfg{rfConfig_t()};
  tmos_memset(&cfg, 0, sizeof(cfg));
  auto task_id{uint8_t(0)};
  task_id = TMOS_ProcessEventRegister(RF_ProcessEvent);
  cfg.accessAddress = 0x71764129;
  cfg.CRCInit = 0x555555;
  cfg.Channel = 39;
  cfg.Frequency = 2480000;
  cfg.LLEMode = LLE_MODE_BASIC | LLE_MODE_EX_CHANNEL;
  cfg.rfStatusCB = RF_2G4StatusCallback;
  cfg.RxMaxlen = 251;
  auto state{RF_Config(&cfg)};
}

int main() {
  // Enable DCDC

  PWR_DCDCCfg(ENABLE);
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  board_button_init();
  board_led_init();
  // I think that fixes the gpio pins to prevent supply voltage from fluctuating

  GPIOA_ModeCfg(GPIO_Pin_All, GPIO_ModeIN_PU);
  GPIOB_ModeCfg(GPIO_Pin_All, GPIO_ModeIN_PU);
  CH59x_BLEInit();
  HAL_Init();
  RF_RoleInit();
  RF_Init();
  return 0;
}
