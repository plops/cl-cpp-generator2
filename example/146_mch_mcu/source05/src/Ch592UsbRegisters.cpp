// no preamble

//

#include "Uart.h"
#include <CH592SFR.h>
#include <array>
extern __attribute((aligned(4))) std::array<uint8_t, 192> EP0_Databuf;
extern __attribute((aligned(4))) std::array<uint8_t, 128> EP1_Databuf;
extern __attribute((aligned(4))) std::array<uint8_t, 128> EP2_Databuf;
extern __attribute((aligned(4))) std::array<uint8_t, 128> EP3_Databuf;
#include "Ch592UsbRegisters.h"
Ch592UsbRegisters::Ch592UsbRegisters() {}
void Ch592UsbRegisters::device_init(uint16_t ep0_data) {
  auto &u{Uart::getInstance()};
  // the following message takes 47us at 6Mbps (actually 7.4Mbps)

  u.print("Usb device_init ep0_data=0x{:X}", ep0_data);
  // Reset control register, clear all settings

  ctrl.reg = 0;
  // Enable Endpoints 4 (OUT+IN) and 1 (OUT+IN)

  ep4_1_mod.ep4_rx_en = 1;
  ep4_1_mod.ep4_tx_en = 1;
  ep4_1_mod.ep1_rx_en = 1;
  ep4_1_mod.ep1_tx_en = 1;
  // Enable Endpoints 2 (OUT+IN) and 3 (OUT+IN)

  ep2_3_mod.ep2_rx_en = 1;
  ep2_3_mod.ep2_tx_en = 1;
  ep2_3_mod.ep3_rx_en = 1;
  ep2_3_mod.ep3_tx_en = 1;
  // Set DMA addresses for Endpoints 0, 1, 2 and 3

  ep0_dma.dma =
      static_cast<uint16_t>(reinterpret_cast<uint32_t>(EP0_Databuf.data()));
  ep1_dma.dma =
      static_cast<uint16_t>(reinterpret_cast<uint32_t>(EP1_Databuf.data()));
  ep2_dma.dma =
      static_cast<uint16_t>(reinterpret_cast<uint32_t>(EP2_Databuf.data()));
  ep3_dma.dma =
      static_cast<uint16_t>(reinterpret_cast<uint32_t>(EP3_Databuf.data()));
  // Configure endpoints, enable automatic ACK on receiving data, and initial
  // NAK on transmitting data

  ep0_ctrl.t_res = 0b10;
  ep1_ctrl.auto_tog = 1;
  ep1_ctrl.t_res = 0b10;
  ep2_ctrl.auto_tog = 1;
  ep2_ctrl.t_res = 0b10;
  ep3_ctrl.auto_tog = 1;
  ep3_ctrl.t_res = 0b10;
  // clear device address

  dev_ad.reg = 0;
  // Enable usb device pull-up resistor, DMA, and interrupts

  ctrl.dma_en = 1;
  ctrl.int_busy = 1;
  ctrl.sys_ctrl = 0b10;
  ctrl.low_speed = 0;
  ctrl.host_mode = 0;
  // Disable analog features on USB pins

  R16_PIN_ANALOG_IE = R16_PIN_ANALOG_IE | RB_PIN_USB_IE | RB_PIN_USB_DP_PU;
  // Clear interrupt flags

  int_flag.reg = 0;
  // Power on the USB port

  port_ctrl.port_en = 1;
  port_ctrl.pd_dis = 1;
  port_ctrl.low_speed = 0;
  port_ctrl.hub0_reset = 0;
  // Enable interrupts for suspend, bus reset, and data transfers

  int_en.suspend = 1;
  int_en.transfer = 1;
  int_en.bus_reset = 1;
}