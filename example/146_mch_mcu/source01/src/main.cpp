extern "C" {
#include <CH59x_common.h>
};
#include <cstdio>

int main() {
  SetSysClock(CLK_SOURCE_PLL_60MHz);
  return 0;
}
