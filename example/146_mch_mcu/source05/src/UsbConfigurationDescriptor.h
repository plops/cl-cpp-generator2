#ifndef USBCONFIGURATIONDESCRIPTOR_H
#define USBCONFIGURATIONDESCRIPTOR_H

#include <cstdint>
#include <ostream>
/**
**Configuration Descriptor Summary**

* **Device configurations:** A USB device can have multiple configurations,
although most devices only have one.
* **Configuration details:** The configuration descriptor specifies power
consumption, interfaces, and transfer mode.
* **Configuration selection:** The host selects a configuration using a
`SetConfiguration` command.

**Descriptor Fields Explained**


| Field               | Description |
|---------------------|------------------------------------------------------------------------------|
| bLength             | Size of the descriptor in bytes. | | bDescriptorType |
Constant value indicating a configuration descriptor (0x02).                 |
| wTotalLength        | Total length in bytes of data returned, including all
following descriptors. | | bNumInterfaces      | Number of interfaces included
in the configuration.                          | | bConfigurationValue | Value
used to select this configuration.                                     | |
iConfiguration      | Index of a string descriptor describing the configuration.
| | bmAttributes        | Bitmap containing power configuration details see
below                      | | bMaxPower           | Maximum power consumption
from the bus in 2mA units (maximum of 500mA).      |

bmAttributes:
    * D7: Reserved (set to 1 for USB 1.0 bus-powered devices).
    * D6: Self-powered.
    * D5: Remote wakeup capable.
    * D4..0: Reserved (set to 0).

I think string descriptors are optional, so for now I will always keep string
indices 0.



*/
class UsbConfigurationDescriptor {
public:
  explicit UsbConfigurationDescriptor(uint8_t b_length_,
                                      uint16_t w_total_length_,
                                      uint8_t b_num_interfaces_,
                                      uint8_t b_configuration_value_,
                                      uint8_t bm_attributes_,
                                      uint8_t b_max_power_);
  bool isValid() const;
  uint8_t GetBLength() const;
  const uint8_t GetBDescriptorType() const;
  uint16_t GetWTotalLength() const;
  uint8_t GetBNumInterfaces() const;
  uint8_t GetBConfigurationValue() const;
  const uint8_t GetIConfiguration() const;
  uint8_t GetBmAttributes() const;
  uint8_t GetBMaxPower() const;

private:
  uint8_t b_length;
  const uint8_t b_descriptor_type{2};
  uint16_t w_total_length;
  uint8_t b_num_interfaces;
  uint8_t b_configuration_value;
  const uint8_t i_configuration{0};
  uint8_t bm_attributes;
  uint8_t b_max_power;
};

#endif /* !USBCONFIGURATIONDESCRIPTOR_H */