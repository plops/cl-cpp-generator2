// no preamble

//

#ifdef BUILD_FOR_TARGET
#define FMT_THROW panic
#endif
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