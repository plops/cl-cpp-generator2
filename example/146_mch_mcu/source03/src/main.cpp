// based on
// https://github.com/openwch/ch592/tree/main/EVT/EXAM/USB/Device/VendorDefinedDev/src

// try to write send data via USB to computer

// AI summary of the example code is here:
// https://github.com/plops/cl-cpp-generator2/tree/master/example/146_mch_mcu/doc/examples/usb/device

extern "C" {
#include <CH59x_common.h>
};
#include <array>
#include <cassert>
constexpr uint16_t DevEP0Size = 0x40;
static_assert(DevEP0Size < 256, "DevEP0Size must fit into one byte.");
// vendor id and product id:

const std::array<uint8_t, 18> DevDescr{0x12, 1,          0x10, 1,    0xFF, 0x80,
                                       0x55, DevEP0Size, 0x48, 0x43, 0x37, 0x55,
                                       0,    1,          1,    2,    0,    1};
const std::array<uint8_t, 74> CfgDescr{
    9, 2,    0x4A, 0,    1,    1,    0,    0x80, 0x32, 9,    4,    0,    0,
    8, 0xFF, 0x80, 0x55, 0,    7,    5,    0x84, 2,    0x40, 0,    0,    7,
    5, 4,    2,    0x40, 0,    0,    7,    5,    0x83, 2,    0x40, 0,    0,
    7, 5,    3,    2,    0x40, 0,    0,    7,    5,    0x82, 2,    0x40, 0,
    0, 7,    5,    2,    2,    0x40, 0,    0,    7,    5,    0x81, 2,    0x40,
    0, 0,    7,    5,    1,    2,    0x40, 0,    0};
const std::array<uint8_t, 4> LangDescr{4, 3, 9, 4};
const std::array<uint8_t, 13> ManuInfo{0xE, 3,   'w', 0,   'c', 0,  'h',
                                       0,   '.', 0,   'c', 0,   'n'};
const std::array<uint8_t, 12> ProdInfo{0xC, 3, 'C', 0, 'H', 0,
                                       '5', 0, '9', 0, 'x', 0};
uint8_t DevConfig;
uint8_t SetupReqCode;
uint16_t SetupReqLen;
const uint8_t *pDescr;
__attribute((aligned(4))) std::array<uint8_t, 192> EP0_Databuf;
__attribute((aligned(4))) std::array<uint8_t, 128> EP1_Databuf;
/** Handle USB transaction processing. Respond to standard USB requests (e.g.
   Get Descriptor, Set Address). Manage data transfers on endpoints.

*/

void USB_DevTransProcess() {
  auto len{uint8_t(0)};
  auto chtype{uint8_t(0)};
  auto errflag{uint8_t(0)};
  auto intflag{uint8_t(R8_USB_INT_FG)};
  if ((intflag & RB_UIF_TRANSFER)) {
    if (!(MASK_UIS_TOKEN == ((R8_USB_INT_ST & MASK_UIS_TOKEN)))) {
      // The following switch extracts the type of token (in/out) and the
      // endpoint number.

      switch ((R8_USB_INT_ST & (MASK_UIS_TOKEN | MASK_UIS_ENDP))) {
      case UIS_TOKEN_IN: {
        switch (SetupReqCode) {
        case USB_GET_DESCRIPTOR: {
          // Handles the standard 'Get Descriptor' request. The device sends the
          // appropriate descriptor data to the host.

          // Calculate length of data to send. Limit to device endpoint size if
          // needed.

          auto new_len{std::min(DevEP0Size, SetupReqLen)};
          assert(new_len < 256);
          len = static_cast<uint8_t>(new_len);
          // Copy the descriptor data to the endpoint buffer for transmission to
          // the host.

          memcpy(pEP0_DataBuf, pDescr, len);
          SetupReqLen -= len;
          pDescr += len;
          // Update state variables (length of the remaining request, pointer to
          // the next chunk of descriptor data) and prepare for the next stage
          // of the transfer.

          R8_UEP0_T_LEN = len;
          R8_UEP0_CTRL = R8_UEP0_CTRL ^ RB_UEP_T_TOG;
          break;
        };
        case USB_SET_ADDRESS: {
          // Handles the standard 'Set Address' request. The device (we) records
          // the new USB address.

          assert(SetupReqLen < 256);
          R8_USB_DEV_AD = (R8_USB_DEV_AD & RB_UDA_GP_BIT) |
                          static_cast<uint8_t>(SetupReqLen);
          R8_UEP0_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
          break;
        };
        default: {
          // Handles any other control requests. This usually results in a stall
          // condition, as the device didn't recognize the request.

          R8_UEP0_T_LEN = 0;
          R8_UEP0_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
          break;
        };
        }
        break;
      };
      case UIS_TOKEN_OUT: {
        // Handles 'OUT' token transactions, meaning the host is sending data to
        // the device.

        // Get length of received data.

        len = R8_USB_RX_LEN;
        break;
      };
      case UIS_TOKEN_OUT | 1: {
        // Handle data reception on endpoint 1.

        if ((R8_USB_INT_ST & RB_UIS_TOG_OK)) {
          // If the data toggle is correct and data is ready.

          // Toggles the receive (IN) data toggle bit for endpoint 1.

          R8_UEP1_CTRL = R8_UEP1_CTRL ^ RB_UEP_R_TOG;
          len = R8_USB_RX_LEN;
          // Get the data length, and call a function (DevEP1_OUT_Deal) to
          // process the received data (on endpoint 1).

          DevEP1_OUT_Deal(len);
        }
        break;
      };
      case UIS_TOKEN_IN | 1: {
        // Prepare an empty (?) response on endpoint 1.

        // Toggle the transmit (OUT) data toggle bit for endpoint 1.

        R8_UEP1_CTRL = R8_UEP1_CTRL ^ RB_UEP_T_TOG;
        // Prepares endpoint 1 for a NAK response (indicating no data is ready
        // to send).

        R8_UEP1_CTRL = (R8_UEP1_CTRL & ~MASK_UEP_T_RES) | UEP_T_RES_NAK;
        break;
      };
      default: {
        break;
      };
      }
      R8_USB_INT_FG = RB_UIF_TRANSFER;
    }
    // This code handles the initial 'Setup' stage of USB control transfers.
    // When the host sends a setup packet to the device, this code analyzes the
    // request and prepares a response.

    if ((R8_USB_INT_ST & RB_UIS_SETUP_ACT)) {
      // A setup packet has been received.

      // Prepare the control endpoint for a response.

      R8_UEP0_CTRL =
          RB_UEP_R_TOG | RB_UEP_T_TOG | UEP_R_RES_ACK | UEP_T_RES_NAK;
      // Extract the length, request code, and type from the setup packet.

      SetupReqLen = pSetupReqPak->wLength;
      SetupReqCode = pSetupReqPak->bRequest;
      chtype = pSetupReqPak->bRequestType;
      len = 0;
      errflag = 0;
      if (USB_REQ_TYP_STANDARD != ((chtype & USB_REQ_TYP_MASK))) {
        // If the request type is NOT a standard request, set an error flag.

        errflag = 0xFF;
      } else {
        // Handle standard request.

        switch (SetupReqCode) {
        case USB_GET_DESCRIPTOR: {
          // Handle requests for device, configuration, or string descriptors.

          switch (pSetupReqPak->wValue >> 8) {
          case USB_DESCR_TYP_DEVICE: {
            pDescr = DevDescr.data();
            len = DevDescr[0];
            break;
          };
          case USB_DESCR_TYP_CONFIG: {
            pDescr = CfgDescr.data();
            len = CfgDescr[2];
            break;
          };
          case USB_DESCR_TYP_STRING: {
            switch ((pSetupReqPak->wValue & 0xFF)) {
            case 0: {
              pDescr = LangDescr.data();
              len = LangDescr.at(0);
              break;
            };
            case 1: {
              pDescr = ManuInfo.data();
              len = ManuInfo.at(0);
              break;
            };
            case 2: {
              pDescr = ProdInfo.data();
              len = ProdInfo.at(0);
              break;
            };
            default: {
              // Unsupported string descriptor type.

              errflag = 0xFF;
              break;
            };
            }
            break;
          };
          default: {
            errflag = 0xFF;
            break;
          };
          }
          // Limit the actual data sent based on the requested length.

          SetupReqLen = std::min(SetupReqLen, static_cast<uint16_t>(len));
          auto new_len{std::min(DevEP0Size, SetupReqLen)};
          assert(new_len < 256);
          len = static_cast<uint8_t>(new_len);
          memcpy(pEP0_DataBuf, pDescr, len);
          pDescr += len;
          break;
        };
        case USB_SET_ADDRESS: {
          SetupReqLen = (pSetupReqPak->wValue & 0xFF);
          break;
        };
        case USB_GET_CONFIGURATION: {
          // Handles the 'Get Configuration' request (responds with the current
          // device configuration).

          // Store configuration in the endpoint buffer for transmission.

          pEP0_DataBuf[0] = DevConfig;
          // Ensure only a single byte is sent (as configuration is one byte).

          SetupReqLen = std::min(static_cast<uint16_t>(1), SetupReqLen);
          break;
        };
        case USB_SET_CONFIGURATION: {
          // Update the DevConfig variable with the new configuration value
          // provided by the host.

          DevConfig = static_cast<uint8_t>((pSetupReqPak->wValue & 0xFF));
          break;
        };
        case USB_CLEAR_FEATURE: {
          // Clear endpoint stalls or other features.

          if (USB_REQ_RECIP_ENDP ==
              ((pSetupReqPak->bRequestType & USB_REQ_RECIP_MASK))) {
            // Request targets an endpoint

            // Clear stall conditions on specific enpoints (number in wIndex).

            switch ((pSetupReqPak->wIndex & 0xFF)) {
            case 0x82: {
              R8_UEP2_CTRL = (R8_UEP2_CTRL & ~(RB_UEP_T_TOG | MASK_UEP_T_RES)) |
                             UEP_T_RES_NAK;
              break;
            };
            case 0x2: {
              R8_UEP2_CTRL = (R8_UEP2_CTRL & ~(RB_UEP_R_TOG | MASK_UEP_R_RES)) |
                             UEP_R_RES_ACK;
              break;
            };
            case 0x81: {
              R8_UEP1_CTRL = (R8_UEP1_CTRL & ~(RB_UEP_T_TOG | MASK_UEP_T_RES)) |
                             UEP_T_RES_NAK;
              break;
            };
            case 0x1: {
              R8_UEP1_CTRL = (R8_UEP1_CTRL & ~(RB_UEP_R_TOG | MASK_UEP_R_RES)) |
                             UEP_R_RES_ACK;
              break;
            };
            default: {
              // Unsupported endpoint number.

              errflag = 0xFF;
              break;
            };
            }
          }
          break;
        };
        case USB_GET_INTERFACE: {
          // Retrieve the alternate setting of the current interface. It seems
          // this device likely only has a single setting (always responds with
          // 0).

          pEP0_DataBuf[0] = 0;
          SetupReqLen = std::min(static_cast<uint16_t>(1), SetupReqLen);
          break;
        };
        case USB_GET_STATUS: {
          // Get device or endpoint status. This implementation only supports a
          // basic status response (all zeros).

          pEP0_DataBuf[0] = 0;
          pEP0_DataBuf[1] = 0;
          SetupReqLen = std::min(static_cast<uint16_t>(2), SetupReqLen);
          break;
        };
        default: {
          // Catch-all for unsupported request codes. Sets an error flag.

          errflag = 0xFF;
          break;
        };
        }
      }
      if (0xFF == errflag) {
        // If the previously set errflag is 0xff (signaling an unsupported
        // request), this code forces a STALL condition on the control endpoint.
        // This signals to the host that the device doesn't recognize the
        // request.

        R8_UEP0_CTRL =
            RB_UEP_R_TOG | RB_UEP_T_TOG | UEP_R_RES_STALL | UEP_T_RES_STALL;
      } else {
        /** Determines Transfer Direction: Checks chtype. If the 0x80 bit is
set, the host expects data from the device (upload/IN direction), otherwise, the
host is sending data (download/OUT direction).

* Sets the data transfer 2length (len) for this stage of the control transfer.

*/
        if ((0x80 & chtype)) {
          // Upload

          auto new_len{std::min(DevEP0Size, SetupReqLen)};
          assert(new_len < 256);
          len = static_cast<uint8_t>(new_len);
          SetupReqLen -= len;
        } else {
          // Download

          len = 0;
        }
        // Configures Endpoint: Prepares the control endpoint register
        // (R8_UEP0_CTRL) for data transmission (likely transitioning to the
        // DATA1 stage of the control transfer).

        R8_UEP0_T_LEN = len;
        R8_UEP0_CTRL =
            RB_UEP_R_TOG | RB_UEP_T_TOG | UEP_R_RES_ACK | UEP_T_RES_ACK;
      }
      // Signals Completion: Sets an interrupt flag (R8_USB_INT_FG =
      // RB_UIF_TRANSFER;) to indicate the setup process is finished.

      R8_USB_INT_FG = RB_UIF_TRANSFER;
    }
  } else if ((intflag & RB_UIF_BUS_RST)) {
    // A bus reset interrupt flag is detected...

    /** 1. Reset Address: Clears the device's address (R8_USB_DEV_AD = 0;),
putting it back to the default address state.
2. Reset Endpoints: Prepares all endpoints (endpoint 0 through 3) for new
transactions.
3. Clear Interrupt Flag: Acknowledges the bus reset interrupt.

*/
    R8_USB_DEV_AD = 0;
    R8_UEP0_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
    R8_UEP1_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
    R8_UEP2_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
    R8_UEP3_CTRL = UEP_R_RES_ACK | UEP_T_RES_NAK;
    R8_USB_INT_FG = RB_UIF_BUS_RST;
  } else if ((intflag & RB_UIF_SUSPEND)) {
    // A suspend interrupt flag is detected...

    /** 1. Check Suspend State: Reads a status register (R8_USB_MIS_ST &
RB_UMS_SUSPEND) to determine if the device is truly suspended.
2. Suspend/Wake-up Actions: The commented sections could contain code to handle
entering a low-power sleep mode (if suspended) or performing wake-up actions (if
resuming from suspend).
3. Clear Interrupt Flag: Acknowledges the suspend interrupt.

*/
    if ((RB_UMS_SUSPEND & R8_USB_MIS_ST)) {
      // Sleep

    } else {
      // Wake up.
    }
    R8_USB_INT_FG = RB_UIF_SUSPEND;
  } else {
    // Catch any other unhandled interrupt flags and simply clears them.

    R8_USB_INT_FG = intflag;
  }
}

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  pEP0_RAM_Addr = EP0_Databuf.data();
  USB_DeviceInit();
  // Enable the interrupt associated with the USB peripheral.

  PFIC_EnableIRQ(USB_IRQn);
  while (1) {
    // inifinite loop
  }
}

/**

__INTERRUPT is defined with __attribute__((interrupt('WCH-Interrupt-fast'))).
This likely indicates a specialized, 'fast' interrupt mechanism specific to your
compiler or microcontroller (WCH).


The compiler attribute __attribute__((section('.highcode'))) will be assigned to
the __HIGH_CODE macro. This attribute likely instructs the compiler to place
functions or code blocks marked with __HIGH_CODE into a special memory section
named '.highcode' (possibly a faster memory region).




*/
__INTERRUPT __HIGH_CODE void USB_IRQHandler() {
  // Handle interrupts coming from the USB Peripheral

  USB_DevTransProcess();
}

void DevEP1_OUT_Deal(uint8_t l) {
  /** Endpoint 1 data reception

1. l Parameter: The argument l represents the length (in bytes) of the received
data packet.

2. Data Inversion: The core of the function is a loop that iterates through each
received byte:

pEP1_IN_DataBuf[i] = ~pEP1_OUT_DataBuf[i]; : This line inverts each byte of data
(~ is the bitwise NOT operator) and stores the result in pEP1_IN_DataBuf.
3. Response Preparation:  The function calls DevEP1_IN_Deal(l).  This other
function is likely responsible for sending the modified data (now in
pEP1_IN_DataBuf) back to the host.


*/
  for (auto i = 0; i < l; i += 1) {
    pEP1_IN_DataBuf[i] = ~(pEP1_OUT_DataBuf[i]);
  }
  DevEP1_IN_Deal(l);
}
