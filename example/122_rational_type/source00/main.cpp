#include <ratio>
#define FMT_HEADER_ONLY
#include "core.h"

float convert_hp(float old_hp, float old_maxhp, float new_maxhp) {
  return ((new_maxhp) * (((old_hp) / (old_maxhp))));
}

int main(int argc, char **argv) {
  fmt::print("generation date 10:21:35 of Saturday, 2023-04-15 (GMT+1)\n");
  auto old_hp = (1.0f);
  auto old_maxhp = (55.f);
  auto new_maxhp = (55.f);
  auto r1over55 = std::ratio<1, 55>;
  auto r55over1 = std::ratio<55, 1>;
  fmt::print("func  convert_hp(old_hp, old_maxhp, new_maxhp)='{}'\n",
             convert_hp(old_hp, old_maxhp, new_maxhp));
  fmt::print("buggy  ((new_maxhp)*(((old_hp)/(old_maxhp))))='{}'\n",
             ((new_maxhp) * (((old_hp) / (old_maxhp)))));
  fmt::print("mul_first  ((((new_maxhp)*(old_hp)))/(old_maxhp))='{}'\n",
             ((((new_maxhp) * (old_hp))) / (old_maxhp)));
  fmt::print("cpp_ratio  std::ratio_add<r55over1,r1over55>='{}'\n",
             std::ratio_add<r55over1, r1over55>);
  fmt::print("lisp_ratio  1='{}'\n", 1);
  fmt::print("lisp_float  (1.00f)='{}'\n", (1.00f));

  return 0;
}
