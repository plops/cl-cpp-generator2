// no preamble

#include "UsbConfigurationDescriptor.h"
#include <sstream>
UsbConfigurationDescriptor::UsbConfigurationDescriptor(
    uint8_t b_length_, uint16_t w_total_length_, uint8_t b_num_interfaces_,
    uint8_t b_configuration_value_, uint8_t bm_attributes_,
    uint8_t b_max_power_)
    : b_length{b_length_}, w_total_length{w_total_length_},
      b_num_interfaces{b_num_interfaces_},
      b_configuration_value{b_configuration_value_},
      bm_attributes{bm_attributes_}, b_max_power{b_max_power_} {}
std::string UsbConfigurationDescriptor::toString() const {
  auto ss{std::ostringstream()};
  ss << "bLength: " << static_cast<int>(b_length) << "\n"
     << "bDescriptorType: " << static_cast<int>(b_descriptor_type) << "\n"
     << "wTotalLength: " << static_cast<int>(w_total_length) << "\n"
     << "bNumInterfaces: " << static_cast<int>(b_num_interfaces) << "\n"
     << "bConfigurationValue: " << static_cast<int>(b_configuration_value)
     << "\n"
     << "iConfiguration: " << static_cast<int>(i_configuration) << "\n"
     << "bmAttributes: " << static_cast<int>(bm_attributes) << "\n"
     << "bMaxPower: " << static_cast<int>(b_max_power) << "\n";
  return ss.str();
}
bool UsbConfigurationDescriptor::isValid() const {
  if (2 != b_descriptor_type || 0 != i_configuration) {
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
const uint8_t UsbConfigurationDescriptor::GetIConfiguration() const {
  return i_configuration;
}
uint8_t UsbConfigurationDescriptor::GetBmAttributes() const {
  return bm_attributes;
}
uint8_t UsbConfigurationDescriptor::GetBMaxPower() const { return b_max_power; }