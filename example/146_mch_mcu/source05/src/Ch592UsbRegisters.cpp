// no preamble

//

#include "Ch592UsbRegisters.h"
#include "Uart.h"
Ch592UsbRegisters::Ch592UsbRegisters() {}
void Ch592UsbRegisters::device_init(uint16_t ep0_data) {
  auto &u{Uart::getInstance()};
  u.print("Usb device_init ep0_data={}", ep0_data);
  ctrl.reg = 0;
  ep4_1_mod.ep4_rx_en = 0;
  ep4_1_mod.ep4_tx_en = 0;
  ep4_1_mod.ep1_rx_en = 0;
  ep4_1_mod.ep1_tx_en = 0;
  ep2_3_mod.ep2_rx_en = 0;
  ep2_3_mod.ep2_tx_en = 0;
  ep2_3_mod.ep3_rx_en = 0;
  ep2_3_mod.ep3_tx_en = 0;
  ep0_dma.dma = ep0_data;
  ep0_ctrl.t_res = 0b11;
}