// no preamble

//

#include "UsbDeviceDescriptor.h"
#include <sstream>
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
  auto ss{std::ostringstream()};
  ss << "bLength: " << std::dec << static_cast<int>(b_length) << " = 0x"
     << std::hex << static_cast<int>(b_length) << "\n"
     << "bDescriptorType: " << std::dec << static_cast<int>(b_descriptor_type)
     << " = 0x" << std::hex << static_cast<int>(b_descriptor_type) << "\n"
     << "bcdUSB: " << std::dec << static_cast<int>(bcd_usb) << " = 0x"
     << std::hex << static_cast<int>(bcd_usb) << "\n"
     << "bDeviceClass: " << std::dec << static_cast<int>(b_device_class)
     << " = 0x" << std::hex << static_cast<int>(b_device_class) << "\n"
     << "bDeviceSubClass: " << std::dec << static_cast<int>(b_device_sub_class)
     << " = 0x" << std::hex << static_cast<int>(b_device_sub_class) << "\n"
     << "bDeviceProtocol: " << std::dec << static_cast<int>(b_device_protocol)
     << " = 0x" << std::hex << static_cast<int>(b_device_protocol) << "\n"
     << "bMaxPacketSize: " << std::dec << static_cast<int>(b_max_packet_size)
     << " = 0x" << std::hex << static_cast<int>(b_max_packet_size) << "\n"
     << "idVendor: " << std::dec << static_cast<int>(id_vendor) << " = 0x"
     << std::hex << static_cast<int>(id_vendor) << "\n"
     << "idProduct: " << std::dec << static_cast<int>(id_product) << " = 0x"
     << std::hex << static_cast<int>(id_product) << "\n"
     << "bcdDevice: " << std::dec << static_cast<int>(bcd_device) << " = 0x"
     << std::hex << static_cast<int>(bcd_device) << "\n"
     << "iManufacturer: " << std::dec << static_cast<int>(i_manufacturer)
     << " = 0x" << std::hex << static_cast<int>(i_manufacturer) << "\n"
     << "iProduct: " << std::dec << static_cast<int>(i_product) << " = 0x"
     << std::hex << static_cast<int>(i_product) << "\n"
     << "iSerialNumber: " << std::dec << static_cast<int>(i_serial_number)
     << " = 0x" << std::hex << static_cast<int>(i_serial_number) << "\n"
     << "bNumConfigurations: " << std::dec
     << static_cast<int>(b_num_configurations) << " = 0x" << std::hex
     << static_cast<int>(b_num_configurations) << "\n";
  return ss.str();
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