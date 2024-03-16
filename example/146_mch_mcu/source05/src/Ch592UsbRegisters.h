#ifndef CH592USBREGISTERS_H
#define CH592USBREGISTERS_H

#include <cstdint>
#include <format.h>
#include <string>
/** @brief The DiagramBase class represents a base class for diagrams.

# Description of Interrupt status register

- MASK_UIS_TOKEN identifies the token PID in USB device mode:
  - 00: OUT packet
  - 01: SOF packet
  - 10: IN packet
  - 11: Idle
- When MASK_UIS_TOKEN is not idle and RB_UIS_SETUP_ACT is 1:
  - Process MASK_UIS_TOKEN first
  - Clear RB_UIF_TRANSFER to make it idle
  - Then process RB_UIS_SETUP_ACT
  - Finally, clear RB_UIF_TRANSFER again
- MASK_UIS_H_RES is valid only in host mode:
  - For OUT/SETUP token packets from host: PID can be ACK/NAK/STALL or indicate
no response/timeout
  - For IN token packets from host: PID can be data packet PID (DATA0/DATA1) or
handshake packet PID

# Description of USB Device Registers

- USB device mode supports 8 bidirectional endpoints: endpoint0 through
endpoint7.
- Maximum data packet length for each endpoint is 64 bytes.
- Endpoint0: Default endpoint, supports control transmission with a shared
64-byte data buffer for transmission and reception.
- Endpoint1, endpoint2, endpoint3: Each has a transmission endpoint IN and a
reception endpoint OUT with separate 64-byte or double 64-byte data buffers,
supporting bulk, interrupt, and real-time/synchronous transmission.
- Endpoint4, endpoint5, endpoint6, endpoint7: Each has a transmission endpoint
IN and a reception endpoint OUT with separate 64-byte data buffers, supporting
bulk, interrupt, and real-time/synchronous transmission.
- Each endpoint has a control register (R8_UEPn_CTRL) and a transmit length
register (R8_UEPn_T_LEN) for setting synchronization trigger bit, response to
OUT/IN transactions, and length of data to be sent.
- USB bus pull-up resistor can be software-controlled via USB control register
(R8_USB_CTRL) for enabling USB device function; not usable in sleep or
power-down mode.
- In sleep mode, pull-up resistor of DP pin can be enabled via R16_PIN_ANALOG_IE
register without being affected.
- USB protocol processor sets interrupt flag for USB bus reset, suspend, wake-up
events, data sending, or receiving; generates interrupt request if enabled.
- Application can query interrupt flag register (R8_USB_INT_FG) and USB
interrupt state register (R8_USB_INT_ST) for processing events based on endpoint
number (MASK_UIS_ENDP) and transaction token PID (MASK_UIS_TOKEN).
- Synchronization trigger bit (RB_UEP_R_TOG) for OUT transactions ensures data
packet received matches the endpoint; data is discarded if not synchronous.
- RB_UEP_AUTO_TOG option available for automatically flipping synchronization
trigger bit after successful transmission or reception.
- Data to be sent/received is stored in their own buffer; sent data length set
in R8_UEPn_T_LEN, received data length in R8_USB_RX_LEN, distinguishable by
current endpoint number during interrupt.


I decided to use an anonymous struct inside of the union, so that I
can write usb.int_flag.transfer instead of usb.int_flag.bit.transfer.
For maximum portability and to adhere strictly to the C++ standard,
it's better to name these structs and access them through named
members. However, compilers like GCC and Clang do support anonymous
structs and unions in C++ as an extension.

As this is only an experiment and only has to work for this particular
chip and the GCC compiler, I'm willing to loose compatibility for
convenience.



When using bit fields and packing them into uint8_t, you're unlikely
to encounter alignment issues for this particular use case. However,
when dealing with larger structs or different types, be mindful of
alignment and how different compilers may pack bit fields differently.
I think I will have to look at the compiler output of -Wpadding and do
some testing to verify the memory alignment of this class matches the
registers.




*/
class Ch592UsbRegisters {
public:
  struct Ctrl {
    union {
      uint8_t reg; // 40008000;
      struct {
        uint8_t dma_en : 1;    // rw ;
        uint8_t clr_all : 1;   // rw  USB FIFO and interrupt flag clear;
        uint8_t reset_sie : 1; // rw  Software reset USB protocol processor;
        uint8_t int_busy : 1;  // rw  Auto pause;
        uint8_t sys_ctlr : 2;  // rw  host-mode==0: 00..disable usb device
                               // function and disable internal pull-up (can be
        // overridden by dev-pullup-en), 01..enable device
        // fucntion, disable internal pull-up, external
        // pull-up-needed, 1x..enable usb device fucntion
        // and internal 1.5k pull-up, pull-up has priority
        // over pull-down resistor;
        uint8_t low_speed : 1; // rw ;
        uint8_t host_mode : 1; // rw ;
      };
    };

    Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "dma-en: {} = 0x{:X},\nclr-all: {} = 0x{:X},\nreset-sie: "
                     "{} = 0x{:X},\nint-busy: {} = 0x{:X},\nsys-ctlr: {} = "
                     "0x{:X},\nlow-speed: {} = 0x{:X},\nhost-mode: {} = 0x{:X}",
                     static_cast<int>(dma_en), static_cast<int>(dma_en),
                     static_cast<int>(clr_all), static_cast<int>(clr_all),
                     static_cast<int>(reset_sie), static_cast<int>(reset_sie),
                     static_cast<int>(int_busy), static_cast<int>(int_busy),
                     static_cast<int>(sys_ctlr), static_cast<int>(sys_ctlr),
                     static_cast<int>(low_speed), static_cast<int>(low_speed),
                     static_cast<int>(host_mode), static_cast<int>(host_mode));
      return out;
    }

  } ctrl{0};
  struct PortCtrl {
    union {
      uint8_t reg; // 40008001;
      struct {
        uint8_t port_en : 1;    // rw  enable USB physical port (disabled
                                // automatically when device detached);
        uint8_t hub0_reset : 1; // rw  0=normal 1=force bus reset;
        uint8_t low_speed : 1;  // rw  enable USB port low speed (0==full speed,
                                // 1== low speed);
        uint8_t reserved3 : 1;  // ro ;
        uint8_t dm_pin : 1;     // ro  UDM pin level;
        uint8_t dp_pin : 1;     // ro  UDP pin level;
        uint8_t reserved6 : 1;  // ro ;
        uint8_t pd_dis : 1;     // rw  disable USB-UDP-UDM pulldown resistance;
      };
    };

    PortCtrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "port-en: {} = 0x{:X},\nhub0-reset: {} = "
                     "0x{:X},\nlow-speed: {} = 0x{:X},\ndm-pin (ro): {} = "
                     "0x{:X},\ndp-pin (ro): {} = 0x{:X},\npd-dis: {} = 0x{:X}",
                     static_cast<int>(port_en), static_cast<int>(port_en),
                     static_cast<int>(hub0_reset), static_cast<int>(hub0_reset),
                     static_cast<int>(low_speed), static_cast<int>(low_speed),
                     static_cast<int>(dm_pin), static_cast<int>(dm_pin),
                     static_cast<int>(dp_pin), static_cast<int>(dp_pin),
                     static_cast<int>(pd_dis), static_cast<int>(pd_dis));
      return out;
    }

  } port_ctrl{0};
  struct IntEn {
    union {
      uint8_t reg; // 40008002;
      struct {
        uint8_t bus_reset : 1; // rw  in USB device mode USB bus reset event
                               // interrupt;
        uint8_t transfer : 1;  // rw  USB transfer completion interrupt;
        uint8_t suspend : 1;  // rw  USB bus suspend or wake-up event interrupt;
        uint8_t host_sof : 1; // rw  host start of frame timing interrupt;
        uint8_t fifo_overflow : 1; // rw  Fifo overflow interrupt;
        uint8_t mod_1_wire_en : 1; // rw  USB single line mode enable;
        uint8_t dev_nak : 1;       // rw  in device mode receive NAK interrupt;
        uint8_t dev_sof : 1; // rw  in device mode receive start of frame (SOF)
                             // packet interrupt;
      };
    };

    IntEn &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(
          std::back_inserter(out),
          "bus-reset: {} = 0x{:X},\ntransfer: {} = 0x{:X},\nsuspend: {} = "
          "0x{:X},\nhost-sof: {} = 0x{:X},\nfifo-overflow: {} = "
          "0x{:X},\nmod-1-wire-en: {} = 0x{:X},\ndev-nak: {} = "
          "0x{:X},\ndev-sof: {} = 0x{:X}",
          static_cast<int>(bus_reset), static_cast<int>(bus_reset),
          static_cast<int>(transfer), static_cast<int>(transfer),
          static_cast<int>(suspend), static_cast<int>(suspend),
          static_cast<int>(host_sof), static_cast<int>(host_sof),
          static_cast<int>(fifo_overflow), static_cast<int>(fifo_overflow),
          static_cast<int>(mod_1_wire_en), static_cast<int>(mod_1_wire_en),
          static_cast<int>(dev_nak), static_cast<int>(dev_nak),
          static_cast<int>(dev_sof), static_cast<int>(dev_sof));
      return out;
    }

  } int_en{0};
  struct DevAd {
    union {
      uint8_t reg; // 40008003;
      struct {
        uint8_t usb_addr : 7; // rw  device mode: the address of the USB itself;
        uint8_t gp_bit : 1;   // rw  USB general flag, user-defined;
      };
    };

    DevAd &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "usb-addr: {} = 0x{:X},\ngp-bit: {} = 0x{:X}",
                     static_cast<int>(usb_addr), static_cast<int>(usb_addr),
                     static_cast<int>(gp_bit), static_cast<int>(gp_bit));
      return out;
    }

  } dev_ad{0};
  struct MiscStatus {
    union {
      uint8_t reg; // 40008005;
      struct {
        uint8_t
            dev_attach : 1;   // ro  USB device connection status of the port in
                              // USB host mode (1 == port has been connected);
        uint8_t dm_level : 1; // ro  In USB host mode, the level status of data
                              // minus (D-, DM) pin when the device is just
                              // connected to the USB port. used to determine
                              // speed (high level, = low speed);
        uint8_t
            bus_suspend : 1; // ro  USB suspend status (is in suspended status);
        uint8_t bus_reset : 1;  // ro  USB bus reset (is at reset status);
        uint8_t r_fifo_rdy : 1; // ro  USB receiver fifo data ready status (not
                                // empty);
        uint8_t sie_free : 1;   // ro  USB proctocol processor free (not busy);
        uint8_t sof_act : 1;    // ro  SOF packet is being sent in host mode;
        uint8_t sof_pre : 1;    // ro  SOF packet will be sent in host mode;
      };
    };

    MiscStatus &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(
          std::back_inserter(out),
          "dev-attach (ro): {} = 0x{:X},\ndm-level (ro): {} = "
          "0x{:X},\nbus-suspend (ro): {} = 0x{:X},\nbus-reset (ro): {} = "
          "0x{:X},\nr-fifo-rdy (ro): {} = 0x{:X},\nsie-free (ro): {} = "
          "0x{:X},\nsof-act (ro): {} = 0x{:X},\nsof-pre (ro): {} = 0x{:X}",
          static_cast<int>(dev_attach), static_cast<int>(dev_attach),
          static_cast<int>(dm_level), static_cast<int>(dm_level),
          static_cast<int>(bus_suspend), static_cast<int>(bus_suspend),
          static_cast<int>(bus_reset), static_cast<int>(bus_reset),
          static_cast<int>(r_fifo_rdy), static_cast<int>(r_fifo_rdy),
          static_cast<int>(sie_free), static_cast<int>(sie_free),
          static_cast<int>(sof_act), static_cast<int>(sof_act),
          static_cast<int>(sof_pre), static_cast<int>(sof_pre));
      return out;
    }

  } misc_status{0};
  struct IntFlag {
    union {
      uint8_t reg; // 40008006;
      struct {
        uint8_t bus_reset : 1; // rw  in device mode: bus reset event trigger.
                               // Write 1 to reset.;
        uint8_t transfer : 1;  // rw  USB transmission completion trigger. Write
                               // 1 to reset.;
        uint8_t suspend : 1; // rw  USB suspend or wake-up event trigger. Write
                             // 1 to reset.;
        uint8_t hst_sof : 1; // rw  SOF packet transmission completion trigger
                             // in USB host mode. Write 1 to reset.;
        uint8_t fifo_ov : 1; // rw  USB FIFO overflow interrupt flag. Write 1 to
                             // reset;
        uint8_t sie_free : 1; // ro  USB processor is idle;
        uint8_t tog_ok : 1; // ro  USB transmission data synchronous flag match
                            // status (1==synchronous, 0==asynchronous);
        uint8_t is_nak : 1; // ro  in device mode: NAK acknowledge during
                            // current USB transmission;
      };
    };

    IntFlag &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(
          std::back_inserter(out),
          "bus-reset: {} = 0x{:X},\ntransfer: {} = 0x{:X},\nsuspend: {} = "
          "0x{:X},\nhst-sof: {} = 0x{:X},\nfifo-ov: {} = 0x{:X},\nsie-free "
          "(ro): {} = 0x{:X},\ntog-ok (ro): {} = 0x{:X},\nis-nak (ro): {} = "
          "0x{:X}",
          static_cast<int>(bus_reset), static_cast<int>(bus_reset),
          static_cast<int>(transfer), static_cast<int>(transfer),
          static_cast<int>(suspend), static_cast<int>(suspend),
          static_cast<int>(hst_sof), static_cast<int>(hst_sof),
          static_cast<int>(fifo_ov), static_cast<int>(fifo_ov),
          static_cast<int>(sie_free), static_cast<int>(sie_free),
          static_cast<int>(tog_ok), static_cast<int>(tog_ok),
          static_cast<int>(is_nak), static_cast<int>(is_nak));
      return out;
    }

  } int_flag{0};
  struct IntStatus {
    union {
      uint8_t reg; // 40008007;
      struct {
        uint8_t endp : 4;   // ro  in device mode the endpoint number of the
                            // current usb transfer transaction;
        uint8_t token : 2;  // ro  in device mode the token pid of the current
                            // usb transfer transaction;
        uint8_t tog_ok : 1; // ro  current usb transmission sync flag matching
                            // status (same as RB_U_TOG_OK), 1=>sync;
        uint8_t setup_act : 1; // ro  in device mode, when this bit is 1, 8-byte
                               // setup request packet has been successfully
                               // received.;
      };
    };

    IntStatus &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(
          std::back_inserter(out),
          "endp (ro): {} = 0x{:X},\ntoken (ro): {} = 0x{:X},\ntog-ok (ro): {} "
          "= 0x{:X},\nsetup-act (ro): {} = 0x{:X}",
          static_cast<int>(endp), static_cast<int>(endp),
          static_cast<int>(token), static_cast<int>(token),
          static_cast<int>(tog_ok), static_cast<int>(tog_ok),
          static_cast<int>(setup_act), static_cast<int>(setup_act));
      return out;
    }

  } int_status{0};
  struct RxLen {
    union {
      uint8_t reg; // 40008008;
      struct {
        uint8_t reserved7 : 1; // ro ;
        uint8_t len : 7; // ro  number of data bytes received by the current usb
                         // endpoint;
      };
    };

    RxLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "len (ro): {} = 0x{:X}",
                     static_cast<int>(len), static_cast<int>(len));
      return out;
    }

  } rx_len{0};
  uint8_t reserved8009{0};
  uint8_t reserved800a{0};
  uint8_t reserved800b{0};
  struct Ep4_1Mod {
    union {
      uint8_t reg; // 4000800C;
      struct {
        uint8_t reserved76 : 2;  // ro ;
        uint8_t ep4_tx_en : 1;   // rw  enable endpoint 4 transmittal (IN);
        uint8_t ep4_rx_en : 1;   // rw  enable endpoint 4 receiving (OUT);
        uint8_t ep1_buf_mod : 1; // rw  endpoint 1 buffer mode;
        uint8_t reserved2 : 1;   // ro ;
        uint8_t ep1_tx_en : 1;   // rw  enable endpoint 1 transmittal (IN);
        uint8_t ep1_rx_en : 1;   // rw  enable endpoint 1 receiving (OUT);
      };
    };

    Ep4_1Mod &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(
          std::back_inserter(out),
          "ep4-tx-en: {} = 0x{:X},\nep4-rx-en: {} = 0x{:X},\nep1-buf-mod: {} = "
          "0x{:X},\nep1-tx-en: {} = 0x{:X},\nep1-rx-en: {} = 0x{:X}",
          static_cast<int>(ep4_tx_en), static_cast<int>(ep4_tx_en),
          static_cast<int>(ep4_rx_en), static_cast<int>(ep4_rx_en),
          static_cast<int>(ep1_buf_mod), static_cast<int>(ep1_buf_mod),
          static_cast<int>(ep1_tx_en), static_cast<int>(ep1_tx_en),
          static_cast<int>(ep1_rx_en), static_cast<int>(ep1_rx_en));
      return out;
    }

  } ep4_1_mod{0};
  struct Ep2_3Mod {
    union {
      uint8_t reg; // 4000800D;
      struct {
        uint8_t ep2_buf_mod : 1; // rw ;
        uint8_t reserved6 : 1;   // ro ;
        uint8_t ep2_tx_en : 1;   // rw ;
        uint8_t ep2_rx_en : 1;   // rw ;
        uint8_t ep3_buf_mod : 1; // rw ;
        uint8_t reserved2 : 1;   // ro ;
        uint8_t ep3_tx_en : 1;   // rw ;
        uint8_t ep3_rx_en : 1;   // rw ;
      };
    };

    Ep2_3Mod &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "ep2-buf-mod: {} = 0x{:X},\nep2-tx-en: {} = "
                     "0x{:X},\nep2-rx-en: {} = 0x{:X},\nep3-buf-mod: {} = "
                     "0x{:X},\nep3-tx-en: {} = 0x{:X},\nep3-rx-en: {} = 0x{:X}",
                     static_cast<int>(ep2_buf_mod),
                     static_cast<int>(ep2_buf_mod), static_cast<int>(ep2_tx_en),
                     static_cast<int>(ep2_tx_en), static_cast<int>(ep2_rx_en),
                     static_cast<int>(ep2_rx_en), static_cast<int>(ep3_buf_mod),
                     static_cast<int>(ep3_buf_mod), static_cast<int>(ep3_tx_en),
                     static_cast<int>(ep3_tx_en), static_cast<int>(ep3_rx_en),
                     static_cast<int>(ep3_rx_en));
      return out;
    }

  } ep2_3_mod{0};
  struct Ep567Mod {
    union {
      uint8_t reg; // 4000800E;
      struct {
        uint8_t ep5_tx_en : 1;  // rw ;
        uint8_t ep5_rx_en : 1;  // rw ;
        uint8_t ep6_tx_en : 1;  // rw ;
        uint8_t ep6_rx_en : 1;  // rw ;
        uint8_t ep7_tx_en : 1;  // rw ;
        uint8_t ep7_rx_en : 1;  // rw ;
        uint8_t reserved01 : 2; // ro ;
      };
    };

    Ep567Mod &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "ep5-tx-en: {} = 0x{:X},\nep5-rx-en: {} = "
                     "0x{:X},\nep6-tx-en: {} = 0x{:X},\nep6-rx-en: {} = "
                     "0x{:X},\nep7-tx-en: {} = 0x{:X},\nep7-rx-en: {} = 0x{:X}",
                     static_cast<int>(ep5_tx_en), static_cast<int>(ep5_tx_en),
                     static_cast<int>(ep5_rx_en), static_cast<int>(ep5_rx_en),
                     static_cast<int>(ep6_tx_en), static_cast<int>(ep6_tx_en),
                     static_cast<int>(ep6_rx_en), static_cast<int>(ep6_rx_en),
                     static_cast<int>(ep7_tx_en), static_cast<int>(ep7_tx_en),
                     static_cast<int>(ep7_rx_en), static_cast<int>(ep7_rx_en));
      return out;
    }

  } ep567_mod{0};
  uint8_t reserved800f{0};
  struct Ep0Dma {
    union {
      uint16_t reg; // 40008010;
      struct {
        uint16_t reserved1514 : 2; // ro ;
        uint16_t dma : 13;         // rw ;
        uint16_t reserved0 : 1;    // ro ;
      };
    };

    Ep0Dma &operator=(uint16_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "dma: {} = 0x{:X}",
                     static_cast<int>(dma), static_cast<int>(dma));
      return out;
    }

  } ep0_dma{0};
  uint16_t reserved40008012{0};
  struct Ep1Dma {
    union {
      uint16_t reg; // 40008014;
      struct {
        uint16_t reserved1514 : 2; // ro ;
        uint16_t dma : 13;         // rw ;
        uint16_t reserved0 : 1;    // ro ;
      };
    };

    Ep1Dma &operator=(uint16_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "dma: {} = 0x{:X}",
                     static_cast<int>(dma), static_cast<int>(dma));
      return out;
    }

  } ep1_dma{0};
  uint16_t reserved40008016{0};
  struct Ep2Dma {
    union {
      uint16_t reg; // 40008018;
      struct {
        uint16_t reserved1514 : 2; // ro ;
        uint16_t dma : 13;         // rw ;
        uint16_t reserved0 : 1;    // ro ;
      };
    };

    Ep2Dma &operator=(uint16_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "dma: {} = 0x{:X}",
                     static_cast<int>(dma), static_cast<int>(dma));
      return out;
    }

  } ep2_dma{0};
  uint16_t reserved4000801_a{0};
  struct Ep3Dma {
    union {
      uint16_t reg; // 4000801C;
      struct {
        uint16_t reserved1514 : 2; // ro ;
        uint16_t dma : 13;         // rw ;
        uint16_t reserved0 : 1;    // ro ;
      };
    };

    Ep3Dma &operator=(uint16_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "dma: {} = 0x{:X}",
                     static_cast<int>(dma), static_cast<int>(dma));
      return out;
    }

  } ep3_dma{0};
  uint16_t reserved4000801_e{0};
  struct Ep0TLen {
    union {
      uint8_t reg; // 40008020;
      struct {
        uint8_t reserved0 : 1; // ro ;
        uint8_t t_len : 7;     // rw  transmit length;
      };
    };

    Ep0TLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "t-len: {} = 0x{:X}",
                     static_cast<int>(t_len), static_cast<int>(t_len));
      return out;
    }

  } ep0_t_len{0};
  uint8_t reserved40008021{0};
  struct Ep0Ctrl {
    union {
      uint8_t reg; // 40008022;
      struct {
        uint8_t
            t_res : 2; // rw  bitmask for of handshake response type for usb
                       // endpoint X, transmittal (in) (see datasheet p. 134);
        uint8_t r_res : 2; // rw  bitmask for of handshake response type for usb
                           // endpoint X, receiving (out);
        uint8_t auto_tog : 1;  // rw  automatic toggle after successful transfer
                               // completion of on of endpoints 1, 2 or 3;
        uint8_t reserved2 : 1; // ro ;
        uint8_t t_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // transmittal (IN), 0=DATA0, 1=DATA1 ;
        uint8_t r_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // receiving (OUT), 0=DATA0, 1=DATA1 ;
      };
    };

    Ep0Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "t-res: {} = 0x{:X},\nr-res: {} = 0x{:X},\nauto-tog: {} = "
                     "0x{:X},\nt-tog: {} = 0x{:X},\nr-tog: {} = 0x{:X}",
                     static_cast<int>(t_res), static_cast<int>(t_res),
                     static_cast<int>(r_res), static_cast<int>(r_res),
                     static_cast<int>(auto_tog), static_cast<int>(auto_tog),
                     static_cast<int>(t_tog), static_cast<int>(t_tog),
                     static_cast<int>(r_tog), static_cast<int>(r_tog));
      return out;
    }

  } ep0_ctrl{0};
  uint8_t reserved40008023{0};
  struct Ep1TLen {
    union {
      uint8_t reg; // 40008024;
      struct {
        uint8_t reserved0 : 1; // ro ;
        uint8_t t_len : 7;     // rw  transmit length;
      };
    };

    Ep1TLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "t-len: {} = 0x{:X}",
                     static_cast<int>(t_len), static_cast<int>(t_len));
      return out;
    }

  } ep1_t_len{0};
  uint8_t reserved40008025{0};
  struct Ep1Ctrl {
    union {
      uint8_t reg; // 40008026;
      struct {
        uint8_t
            t_res : 2; // rw  bitmask for of handshake response type for usb
                       // endpoint X, transmittal (in) (see datasheet p. 134);
        uint8_t r_res : 2; // rw  bitmask for of handshake response type for usb
                           // endpoint X, receiving (out);
        uint8_t auto_tog : 1;  // rw  automatic toggle after successful transfer
                               // completion of on of endpoints 1, 2 or 3;
        uint8_t reserved2 : 1; // ro ;
        uint8_t t_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // transmittal (IN), 0=DATA0, 1=DATA1 ;
        uint8_t r_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // receiving (OUT), 0=DATA0, 1=DATA1 ;
      };
    };

    Ep1Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "t-res: {} = 0x{:X},\nr-res: {} = 0x{:X},\nauto-tog: {} = "
                     "0x{:X},\nt-tog: {} = 0x{:X},\nr-tog: {} = 0x{:X}",
                     static_cast<int>(t_res), static_cast<int>(t_res),
                     static_cast<int>(r_res), static_cast<int>(r_res),
                     static_cast<int>(auto_tog), static_cast<int>(auto_tog),
                     static_cast<int>(t_tog), static_cast<int>(t_tog),
                     static_cast<int>(r_tog), static_cast<int>(r_tog));
      return out;
    }

  } ep1_ctrl{0};
  uint8_t reserved40008027{0};
  struct Ep2TLen {
    union {
      uint8_t reg; // 40008028;
      struct {
        uint8_t reserved0 : 1; // ro ;
        uint8_t t_len : 7;     // rw  transmit length;
      };
    };

    Ep2TLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "t-len: {} = 0x{:X}",
                     static_cast<int>(t_len), static_cast<int>(t_len));
      return out;
    }

  } ep2_t_len{0};
  uint8_t reserved40008029{0};
  struct Ep2Ctrl {
    union {
      uint8_t reg; // 4000802A;
      struct {
        uint8_t
            t_res : 2; // rw  bitmask for of handshake response type for usb
                       // endpoint X, transmittal (in) (see datasheet p. 134);
        uint8_t r_res : 2; // rw  bitmask for of handshake response type for usb
                           // endpoint X, receiving (out);
        uint8_t auto_tog : 1;  // rw  automatic toggle after successful transfer
                               // completion of on of endpoints 1, 2 or 3;
        uint8_t reserved2 : 1; // ro ;
        uint8_t t_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // transmittal (IN), 0=DATA0, 1=DATA1 ;
        uint8_t r_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // receiving (OUT), 0=DATA0, 1=DATA1 ;
      };
    };

    Ep2Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "t-res: {} = 0x{:X},\nr-res: {} = 0x{:X},\nauto-tog: {} = "
                     "0x{:X},\nt-tog: {} = 0x{:X},\nr-tog: {} = 0x{:X}",
                     static_cast<int>(t_res), static_cast<int>(t_res),
                     static_cast<int>(r_res), static_cast<int>(r_res),
                     static_cast<int>(auto_tog), static_cast<int>(auto_tog),
                     static_cast<int>(t_tog), static_cast<int>(t_tog),
                     static_cast<int>(r_tog), static_cast<int>(r_tog));
      return out;
    }

  } ep2_ctrl{0};
  uint8_t reserved4000802_b{0};
  struct Ep3TLen {
    union {
      uint8_t reg; // 4000802C;
      struct {
        uint8_t reserved0 : 1; // ro ;
        uint8_t t_len : 7;     // rw  transmit length;
      };
    };

    Ep3TLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "t-len: {} = 0x{:X}",
                     static_cast<int>(t_len), static_cast<int>(t_len));
      return out;
    }

  } ep3_t_len{0};
  uint8_t reserved4000802_d{0};
  struct Ep3Ctrl {
    union {
      uint8_t reg; // 4000802E;
      struct {
        uint8_t
            t_res : 2; // rw  bitmask for of handshake response type for usb
                       // endpoint X, transmittal (in) (see datasheet p. 134);
        uint8_t r_res : 2; // rw  bitmask for of handshake response type for usb
                           // endpoint X, receiving (out);
        uint8_t auto_tog : 1;  // rw  automatic toggle after successful transfer
                               // completion of on of endpoints 1, 2 or 3;
        uint8_t reserved2 : 1; // ro ;
        uint8_t t_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // transmittal (IN), 0=DATA0, 1=DATA1 ;
        uint8_t r_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // receiving (OUT), 0=DATA0, 1=DATA1 ;
      };
    };

    Ep3Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "t-res: {} = 0x{:X},\nr-res: {} = 0x{:X},\nauto-tog: {} = "
                     "0x{:X},\nt-tog: {} = 0x{:X},\nr-tog: {} = 0x{:X}",
                     static_cast<int>(t_res), static_cast<int>(t_res),
                     static_cast<int>(r_res), static_cast<int>(r_res),
                     static_cast<int>(auto_tog), static_cast<int>(auto_tog),
                     static_cast<int>(t_tog), static_cast<int>(t_tog),
                     static_cast<int>(r_tog), static_cast<int>(r_tog));
      return out;
    }

  } ep3_ctrl{0};
  uint8_t reserved4000802_f{0};
  struct Ep4TLen {
    union {
      uint8_t reg; // 40008030;
      struct {
        uint8_t reserved0 : 1; // ro ;
        uint8_t t_len : 7;     // rw  transmit length;
      };
    };

    Ep4TLen &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out), "t-len: {} = 0x{:X}",
                     static_cast<int>(t_len), static_cast<int>(t_len));
      return out;
    }

  } ep4_t_len{0};
  uint8_t reserved40008031{0};
  struct Ep4Ctrl {
    union {
      uint8_t reg; // 40008032;
      struct {
        uint8_t
            t_res : 2; // rw  bitmask for of handshake response type for usb
                       // endpoint X, transmittal (in) (see datasheet p. 134);
        uint8_t r_res : 2; // rw  bitmask for of handshake response type for usb
                           // endpoint X, receiving (out);
        uint8_t auto_tog : 1;  // rw  automatic toggle after successful transfer
                               // completion of on of endpoints 1, 2 or 3;
        uint8_t reserved2 : 1; // ro ;
        uint8_t t_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // transmittal (IN), 0=DATA0, 1=DATA1 ;
        uint8_t r_tog : 1; // rw  prepared data toggle flag of USB endpoint X
                           // receiving (OUT), 0=DATA0, 1=DATA1 ;
      };
    };

    Ep4Ctrl &operator=(uint8_t value) {
      reg = value;
      return *this;
    }

    std::string toString() const {
      auto out{std::string()};
      fmt::format_to(std::back_inserter(out),
                     "t-res: {} = 0x{:X},\nr-res: {} = 0x{:X},\nauto-tog: {} = "
                     "0x{:X},\nt-tog: {} = 0x{:X},\nr-tog: {} = 0x{:X}",
                     static_cast<int>(t_res), static_cast<int>(t_res),
                     static_cast<int>(r_res), static_cast<int>(r_res),
                     static_cast<int>(auto_tog), static_cast<int>(auto_tog),
                     static_cast<int>(t_tog), static_cast<int>(t_tog),
                     static_cast<int>(r_tog), static_cast<int>(r_tog));
      return out;
    }

  } ep4_ctrl{0};
  uint8_t reserved40008033{0};

public:
  explicit Ch592UsbRegisters();
  void device_init(uint16_t ep0_data);

private:
};

#endif /* !CH592USBREGISTERS_H */