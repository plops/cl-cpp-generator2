// based on
// https://github.com/openwch/ch592/tree/main/EVT/EXAM/USB/Device/VendorDefinedDev/src

// try to write send data via USB to computer

// AI summary of the example code is here:
// https://github.com/plops/cl-cpp-generator2/tree/master/example/146_mch_mcu/doc/examples/usb/device

#include "Ch592UsbRegisters.h"
#include "UsbConfigurationDescriptor.h"
#include "UsbDeviceDescriptor.h"
#include <array>
#include <cassert>
extern "C" {
#include <CH59x_common.h>
};
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
constexpr uintptr_t c_USB_BASE_ADDR = 0x40008000;
Ch592UsbRegisters &usb =
    *new (reinterpret_cast<void *>(c_USB_BASE_ADDR)) Ch592UsbRegisters;
// overview usb https://www.beyondlogic.org/usbnutshell/usb3.shtml

void USB_DevTransProcess2() {
  if (usb.int_flag.transfer) {
    if (usb.int_status.token | usb.int_status.endp) {
      // handle requests
    }
    // clear interrupt by writing to flag

    usb.int_flag.transfer = 1;
  }
}

/** Handle USB transaction processing. Respond to standard USB requests (e.g.
   Get Descriptor, Set Address). Manage data transfers on endpoints.

*/
/**
Here's a bullet list summary of the essential concepts regarding USB Protocols:

**Understanding USB Protocols**

* **Layered Structure:** USB protocols operate in layers, simplifying design.
Higher layers are more relevant for users, with lower layers handled by USB
controllers.
* **Transactions:** USB data exchange occurs in transactions with the following
components:
    * Token Packet (header)
    * Optional Data Packet (payload)
    * Status Packet (acknowledgment/error correction)

**Key Packet Fields:**

* **Sync:** Synchronizes transmitter and receiver clocks.
* **PID (Packet ID):**  Identifies the packet type (token, data, handshake,
etc.).
* **ADDR:** Specifies the destination device address.
* **ENDP:** Identifies the specific endpoint on the device.
* **CRC:** Error detection mechanism.
* **EOP:** Marks the end of a packet.

**Packet Types**

* **Token:** Indicates transaction type (IN, OUT, SETUP)
* **Data:** Carries the actual data payload (DATA0, DATA 1, etc.).
* **Handshake:** Acknowledges transactions or signals errors (ACK, NAK, STALL)
* **Start of Frame (SOF):** Sent periodically to mark time intervals.

**USB Functions and Devices**

* **USB Function:** A USB device with a specific capability (printer, scanner,
etc.). Note that this can include host controllers or hubs as well.
* **Endpoints:** Points on a USB function where data is sent or received.
Endpoint 0 is mandatory for control/status.
* **Pipes:** Logical connections between the host software and device endpoints,
defining transfer parameters.

**Key Points**

* USB is host-centric; the host initiates all transactions.
* Most USB controllers handle low-level protocol implementation.
* Understanding endpoints and pipes is crucial for USB device design.




*/
/**
**Control Transfers: Purpose and Characteristics**

* **Function:** Used for device setup (enumeration), command & status operations
* **Initiation:** Always started by the host computer
* **Nature:** Bursty, random packets
* **Error Handling:** Utilize a best-effort delivery approach
* **Packet Sizes:**
    * Low-speed: 8 bytes
    * Full-speed: 64 bytes
    * High-speed: 8, 16, 32, or 64 bytes

**Stages of a Control Transfer**

1. **Setup Stage:**
   * Host sends a setup token (address, endpoint)
   * Host sends a data packet (DATA0) containing the setup details
   * Device acknowledges (ACK) if data is received successfully

2. **Optional Data Stage:**
   * One or more IN/OUT transactions depending on data direction
   * Data is sent in chunks matching the maximum packet size
   * Device can signal readiness (ACK), temporary unavailability (NAK), or an
error (STALL)

3. **Status Stage:**
    *  Direction dictates the status reporting procedure:
       * IN Transfer: Host acknowledges data, device reports status
       * OUT Transfer: Device acknowledges data, host checks status

**Big Picture Example: Requesting a Device Descriptor**

1. **Setup:** Host sends a setup token, then a DATA0 packet with the descriptor
request. Device acknowledges.
2. **Data:** Host sends IN tokens. Device sends the descriptor in chunks, with
the host acknowledging each chunk.
3. **Status:** Host sends a zero-length OUT packet to signal success, the device
responds to confirm its own status.


*/

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  pEP0_RAM_Addr = EP0_Databuf.data();
  auto &dev{*reinterpret_cast<const UsbDeviceDescriptor *>(DevDescr.data())};
  auto &cfg{
      *reinterpret_cast<const UsbConfigurationDescriptor *>(CfgDescr.data())};
  dev.isValid();
  cfg.isValid();
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

  USB_DevTransProcess2();
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