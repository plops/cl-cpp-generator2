import Vector;
import "cmath";
// Stroustrup Tour of C++ (2022) page 35
// https://www.reddit.com/r/cpp/comments/zswkp8/modules_in_the_big_three_compilers_a_small/

double sqrt_sum(Vector &v) {
  auto sum = 0.;
  for (auto i = 0; i < v.size(); i += 1) {
    sum += v[i];
  }
  return sum;
}

int main(int argc, char **argv) {
  auto v = Vector(3);
  sqrt_sum(v);
}
