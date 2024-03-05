#ifndef USBDEVICEDESCRIPTOR_H
#define USBDEVICEDESCRIPTOR_H

#include <cstdint>
/**
**Device Descriptor**

* **Represents the entire USB device:**  One device = one descriptor.
* **Key Device Information:**
    * **USB Version Supported:**  Device's USB spec compliance (e.g., 2.0, 1.1)
    * **Maximum Packet Size (Endpoint 0):**  Largest data unit for default
endpoint.
    * **Vendor ID:**  USB-IF assigned ID for the device's manufacturer.
    * **Product ID:** Manufacturer-assigned ID for the specific device.
    * **Number of Configurations:** How many ways the device can be configured.

**Understanding Fields**

* **bcdUSB:** Binary-coded decimal for USB version (e.g., 0x0200 = USB 2.0)
* **bDeviceClass, bDeviceSubClass, bDeviceProtocol:**  Codes used to find the
appropriate device driver. Often more specific codes are defined at the
interface level.
* **bcdDevice:** Device version number set by the developer.
* **iManufacturer, iProduct, iSerialNumber:**  Indexes pointing to optional
string descriptors for additional human-readable information.
* **bNumConfigurations:**  Indicates the total number of potential device
setups.


*/
class UsbDeviceDescriptor {
public:
  explicit UsbDeviceDescriptor(uint8_t b_device_class_,
                               uint8_t b_device_sub_class_,
                               uint8_t b_device_protocol_, uint16_t id_vendor_,
                               uint16_t id_product_, uint16_t bcd_device_,
                               uint8_t b_num_configurations_);
  /**
@brief isValid() checks if the const members b_length,
b_descriptor_type, bcd_usb, and b_max_packet_size have the expected
values. These values are defined based on the USB specification that
the UsbDeviceDescriptor is designed to represent. In a real-world
scenario, these checks ensure that the hardcoded values haven't been
tampered with or incorrectly modified due to a programming error or
memory corruption.

This method shall be used if you cast an arbitrary uint8_t array to
UsbDeviceDescriptor.


*/
  bool isValid() const;
  const uint8_t GetBLength() const;
  const uint8_t GetBDescriptorType() const;
  const uint16_t GetBcdUsb() const;
  uint8_t GetBDeviceClass() const;
  uint8_t GetBDeviceSubClass() const;
  uint8_t GetBDeviceProtocol() const;
  const uint8_t GetBMaxPacketSize() const;
  uint16_t GetIdVendor() const;
  uint16_t GetIdProduct() const;
  uint16_t GetBcdDevice() const;
  const uint8_t GetIManufacturer() const;
  const uint8_t GetIProduct() const;
  const uint8_t GetISerialNumber() const;
  uint8_t GetBNumConfigurations() const;

private:
  const uint8_t b_length{18};
  const uint8_t b_descriptor_type{1};
  const uint16_t bcd_usb{0x200};
  uint8_t b_device_class;
  uint8_t b_device_sub_class;
  uint8_t b_device_protocol;
  const uint8_t b_max_packet_size{64};
  uint16_t id_vendor;
  uint16_t id_product;
  uint16_t bcd_device;
  const uint8_t i_manufacturer{0};
  const uint8_t i_product{0};
  const uint8_t i_serial_number{0};
  uint8_t b_num_configurations;
};

#endif /* !USBDEVICEDESCRIPTOR_H */