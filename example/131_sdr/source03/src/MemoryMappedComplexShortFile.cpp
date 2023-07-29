// no preamble

#include <stdexcept>

#include "MemoryMappedComplexShortFile.h"
MemoryMappedComplexShortFile::MemoryMappedComplexShortFile(
    const std::string &filename)
    : filename_(filename) {
  file_.open(filename);
  if (!file_.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  data_ =
      reinterpret_cast<std::complex<short> *>(const_cast<char *>(file_.data()));
}
std::complex<short> &
MemoryMappedComplexShortFile::operator[](std::size_t index) {
  return data_[index];
}
MemoryMappedComplexShortFile::~MemoryMappedComplexShortFile() { file_.close(); }
