Import Vector;
#include <cmath>
// Stroustrup Tour of C++ (2022) page 35

void sqrt_sum(Vector &v) {
  auto sum = 0.;
  for (auto i = 0; i < v.size(); i += 1) {
    sum += std::sqrt(v[i]);
  }
  return sum;
}

int main(int argc, char **argv) {
  std::cout << std::format("main entry point argc='{}' argv[0]='{}'\n", argc,
                           argv[0]);
  sqrt_sum(Vector(3));
}
