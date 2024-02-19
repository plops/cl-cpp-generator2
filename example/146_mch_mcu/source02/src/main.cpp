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
#include <array>
#include <cstdio>
__attribute((aligned(4))) uint32_t MEM_BUF[(BLE_MEMHEAP_SIZE / 4)];
const uint8_t MacAddr[6]{0x84, 0xC2, 0xE4, 3, 2, 2};
std::array<uint8_t, 10> TX_DATA{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
enum class SBP : uint16_t {
  START_DEVICE_EVT,
  SBP_RF_PERIODIC_EVT,
  SBP_RF_RF_RX_EVT
};
__HIGH_CODE __attribute((noinline)) void Main_Circulation() {
  while (true) {
    TMOS_SystemProcess();
  }
}

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
  auto taskID{uint8_t(0)};
  taskID = TMOS_ProcessEventRegister(RF_ProcessEvent);
  cfg.accessAddress = 0x71764129;
  cfg.CRCInit = 0x555555;
  cfg.Channel = 39;
  cfg.Frequency = 2480000;
  cfg.LLEMode = LLE_MODE_BASIC | LLE_MODE_EX_CHANNEL;
  cfg.rfStatusCB = RF_2G4StatusCallback;
  cfg.RxMaxlen = 251;
  auto state{RF_Config(&cfg)};
  if (true) {
    // TX mode

    tmos_set_event(taskID, static_cast<uint16_t>(SBP::SBP_RF_PERIODIC_EVT));
  }
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
  Main_Circulation();
  return 0;
}
