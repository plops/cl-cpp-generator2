// no preamble

//

#include "UsbConfigurationDescriptor.h"
UsbConfigurationDescriptor::UsbConfigurationDescriptor(
    uint8_t b_length_, uint16_t w_total_length_, uint8_t b_num_interfaces_,
    uint8_t b_configuration_value_, uint8_t i_configuration_,
    uint8_t bm_attributes_, uint8_t b_, uint8_t b_max_power_)
    : b_length{b_length_}, w_total_length{w_total_length_},
      b_num_interfaces{b_num_interfaces_},
      b_configuration_value{b_configuration_value_},
      i_configuration{i_configuration_}, bm_attributes{bm_attributes_}, b{b_},
      b_max_power{b_max_power_} {
  static_assert(18 == sizeof(UsbDeviceDescriptor));
}
bool UsbConfigurationDescriptor::isValid() const {
  if (2 != b_descriptor_type) {
    return false;
  }
  return true;
}
uint8_t UsbConfigurationDescriptor::GetBLength() const { return b_length; }
const uint8_t UsbConfigurationDescriptor::GetBDescriptorType() const {
  return b_descriptor_type;
}
uint16_t UsbConfigurationDescriptor::GetWTotalLength() const {
  return w_total_length;
}
uint8_t UsbConfigurationDescriptor::GetBNumInterfaces() const {
  return b_num_interfaces;
}
uint8_t UsbConfigurationDescriptor::GetBConfigurationValue() const {
  return b_configuration_value;
}
uint8_t UsbConfigurationDescriptor::GetIConfiguration() const {
  return i_configuration;
}
uint8_t UsbConfigurationDescriptor::GetBmAttributes() const {
  return bm_attributes;
}
uint8_t UsbConfigurationDescriptor::GetB() const { return b; }
uint8_t UsbConfigurationDescriptor::GetBMaxPower() const { return b_max_power; }