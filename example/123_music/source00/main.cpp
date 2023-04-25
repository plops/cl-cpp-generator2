#include <ratio>
#define FMT_HEADER_ONLY
#include "core.h"

float convert_hp(float old_hp, float old_maxhp, float new_maxhp) {
  return ((new_maxhp) * (((old_hp) / (old_maxhp))));
}

int main(int argc, char **argv) {
  fmt::print("generation date 01:16:42 of Wednesday, 2023-04-26 (GMT+1)\n");
  auto old_hp = (1.0f);
  auto old_maxhp = (55.f);
  auto new_maxhp = (55.f);
  auto dold_hp = (1.0);
  auto dold_maxhp = (55.);
  auto dnew_maxhp = (55.);
  fmt::print("when the computation is <1, this is due to a floating point "
             "rounding error and leads to a x\n");
  fmt::print("func  convert_hp(old_hp, old_maxhp, new_maxhp)='{}'\n",
             convert_hp(old_hp, old_maxhp, new_maxhp));
  fmt::print("buggy  ((new_maxhp)*(((old_hp)/(old_maxhp))))='{}'\n",
             ((new_maxhp) * (((old_hp) / (old_maxhp)))));
  fmt::print("buggy_double  ((dnew_maxhp)*(((dold_hp)/(dold_maxhp))))='{}'\n",
             ((dnew_maxhp) * (((dold_hp) / (dold_maxhp)))));
  fmt::print("mul_first  ((((new_maxhp)*(old_hp)))/(old_maxhp))='{}'\n",
             ((((new_maxhp) * (old_hp))) / (old_maxhp)));
  fmt::print("lisp_ratio  1='{}'\n", 1);
  fmt::print("lisp_float  (1.00f)='{}'\n", (1.00f));

  return 0;
}
