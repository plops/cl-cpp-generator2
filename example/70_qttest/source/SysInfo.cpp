// no preamble
;
#include "SysInfo.h"
#include "SysInfoLinuxImpl.h"
SysInfo &SysInfo::instance() {
  static SysInfoLinuxImpl singleton;
  return singleton;
}
SysInfo::~SysInfo() {}
SysInfo::SysInfo() {}
SysInfo::SysInfo(const SysInfo &rhs) {}
SysInfo &SysInfo::operator=(const SysInfo &rhs) { return *this; }