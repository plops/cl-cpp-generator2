// no preamble

//

#include "UsbDeviceDescriptor.h"
#include <format>
UsbDeviceDescriptor::UsbDeviceDescriptor(
    uint8_t b_device_class_, uint8_t b_device_sub_class_,
    uint8_t b_device_protocol_, uint16_t id_vendor_, uint16_t id_product_,
    uint16_t bcd_device_, uint8_t b_num_configurations_)
    : b_device_class{b_device_class_}, b_device_sub_class{b_device_sub_class_},
      b_device_protocol{b_device_protocol_}, id_vendor{id_vendor_},
      id_product{id_product_}, bcd_device{bcd_device_},
      b_num_configurations{b_num_configurations_} {
  static_assert(18 == sizeof(UsbDeviceDescriptor));
}
std::string UsbDeviceDescriptor::toString() const {
  return std::format(
      "bLength: {} = 0x{:X},\nbDescriptorType: {} = 0x{:X},\nbcdUSB: {} = "
      "0x{:X},\nbDeviceClass: {} = 0x{:X},\nbDeviceSubClass: {} = "
      "0x{:X},\nbDeviceProtocol: {} = 0x{:X},\nbMaxPacketSize: {} = "
      "0x{:X},\nidVendor: {} = 0x{:X},\nidProduct: {} = 0x{:X},\nbcdDevice: {} "
      "= 0x{:X},\niManufacturer: {} = 0x{:X},\niProduct: {} = "
      "0x{:X},\niSerialNumber: {} = 0x{:X},\nbNumConfigurations: {} = 0x{:X}",
      static_cast<int>(b_length), static_cast<int>(b_length),
      static_cast<int>(b_descriptor_type), static_cast<int>(b_descriptor_type),
      static_cast<int>(bcd_usb), static_cast<int>(bcd_usb),
      static_cast<int>(b_device_class), static_cast<int>(b_device_class),
      static_cast<int>(b_device_sub_class),
      static_cast<int>(b_device_sub_class), static_cast<int>(b_device_protocol),
      static_cast<int>(b_device_protocol), static_cast<int>(b_max_packet_size),
      static_cast<int>(b_max_packet_size), static_cast<int>(id_vendor),
      static_cast<int>(id_vendor), static_cast<int>(id_product),
      static_cast<int>(id_product), static_cast<int>(bcd_device),
      static_cast<int>(bcd_device), static_cast<int>(i_manufacturer),
      static_cast<int>(i_manufacturer), static_cast<int>(i_product),
      static_cast<int>(i_product), static_cast<int>(i_serial_number),
      static_cast<int>(i_serial_number), static_cast<int>(b_num_configurations),
      static_cast<int>(b_num_configurations));
}
bool UsbDeviceDescriptor::isValid() const {
  if (18 != b_length || 1 != b_descriptor_type || 0x200 != bcd_usb ||
      64 != b_max_packet_size || 0 != i_manufacturer || 0 != i_product ||
      0 != i_serial_number) {
    return false;
  }
  return true;
}
const uint8_t UsbDeviceDescriptor::GetBLength() const { return b_length; }
const uint8_t UsbDeviceDescriptor::GetBDescriptorType() const {
  return b_descriptor_type;
}
const uint16_t UsbDeviceDescriptor::GetBcdUsb() const { return bcd_usb; }
uint8_t UsbDeviceDescriptor::GetBDeviceClass() const { return b_device_class; }
uint8_t UsbDeviceDescriptor::GetBDeviceSubClass() const {
  return b_device_sub_class;
}
uint8_t UsbDeviceDescriptor::GetBDeviceProtocol() const {
  return b_device_protocol;
}
const uint8_t UsbDeviceDescriptor::GetBMaxPacketSize() const {
  return b_max_packet_size;
}
uint16_t UsbDeviceDescriptor::GetIdVendor() const { return id_vendor; }
uint16_t UsbDeviceDescriptor::GetIdProduct() const { return id_product; }
uint16_t UsbDeviceDescriptor::GetBcdDevice() const { return bcd_device; }
const uint8_t UsbDeviceDescriptor::GetIManufacturer() const {
  return i_manufacturer;
}
const uint8_t UsbDeviceDescriptor::GetIProduct() const { return i_product; }
const uint8_t UsbDeviceDescriptor::GetISerialNumber() const {
  return i_serial_number;
}
uint8_t UsbDeviceDescriptor::GetBNumConfigurations() const {
  return b_num_configurations;
}