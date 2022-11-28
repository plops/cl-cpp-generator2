// no preamble
;
#include "SysInfoLinuxImpl.h"
#include <QFile>
#include <sys/sysinfo.h>
#include <sys/types.h>
QVector<qulonglong> SysInfoLinuxImpl::cpuRawData() {
  QFile file("/proc/stat");
  file.open(QIODevice::ReadOnly);
  auto line = file.readLine();
  file.close();
  auto totalUser = qulonglong(0);
  auto totalUserNice = qulonglong(0);
  auto totalSystem = qulonglong(0);
  auto totalIdle = qulonglong(0);
  std::sscanf(line.data(), "cpu %llu %llu %llu %llu", &(totalUser),
              &(totalUserNice), &(totalSystem), &(totalIdle));
  QVector<qulonglong> rawData;
  rawData.append(totalUser);
  rawData.append(totalUserNice);
  rawData.append(totalSystem);
  rawData.append(totalIdle);
  return rawData;
}
SysInfoLinuxImpl::SysInfoLinuxImpl() : SysInfo(), cpu_load_last_values_() {}
void SysInfoLinuxImpl::init() { cpu_load_last_values_ = cpuRawData(); }
double SysInfoLinuxImpl::cpuLoadAverage() {
  auto firstSample = cpu_load_last_values_;
  auto secondSample = cpuRawData();
  cpu_load_last_values_ = secondSample;
  auto overall = ((((secondSample[0]) - (firstSample[0]))) +
                  (((secondSample[1]) - (firstSample[1]))) +
                  (((secondSample[2]) - (firstSample[2]))));
  auto total = ((overall) + (((secondSample[3]) - (firstSample[3]))));
  auto percent = (((((1.00e+2)) * (overall))) / (total));
  return qBound((0.), percent, (1.00e+2));
}
double SysInfoLinuxImpl::memoryUsed() {
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  auto totalMemory = ((((qulonglong(memInfo.totalram)) + (memInfo.totalswap))) *
                      (memInfo.mem_unit));
  auto totalMemoryUsed =
      ((((qulonglong(((memInfo.totalram) - (memInfo.freeram)))) +
         (((memInfo.totalswap) - (memInfo.freeswap))))) *
       (memInfo.mem_unit));
  auto percent = (((((1.00e+2)) * (totalMemoryUsed))) / (totalMemory));
  return qBound((0.), percent, (1.00e+2));
}