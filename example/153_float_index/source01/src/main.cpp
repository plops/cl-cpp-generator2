#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
// This is a mapping between floating point numbers and integer indices that
// keep order

float index_to_float(uint32_t n) {
  n += ((1u << 23) - 1);
  if (n & (1u << 31)) {
    n = n ^ (1u << 31);
  } else {
    n = ~n;
  }
  float f;
  memcpy(&f, &n, 4);
  return f;
}

uint32_t float_to_index(float f) {
  uint32_t n;
  memcpy(&n, &f, sizeof(n));
  if (n & (1u << 31)) {
    n = ~(n ^ (1u << 31));
  } else {
    n = ~n;
  }
  // Ensure the subtraction is done as unsigned
  return n - ((1u << 31) + ((1u << 23) - 1u));
}

int main(int argc, char **argv) {
  /**

v='0'  index_to_float(v)='-inf'  float_to_index(index_to_float(v))='0'
v='1'  index_to_float(v)='-3.40282e+38'  float_to_index(index_to_float(v))='1'
v='12'  index_to_float(v)='-3.40282e+38'  float_to_index(index_to_float(v))='12'
v='1000'  index_to_float(v)='-3.40262e+38'
float_to_index(index_to_float(v))='1000' v='10000'
index_to_float(v)='-3.4008e+38'  float_to_index(index_to_float(v))='10000'
v='100000'  index_to_float(v)='-3.38254e+38'
float_to_index(index_to_float(v))='100000' v='1000000000'
index_to_float(v)='-458.422'  float_to_index(index_to_float(v))='1000000000'
v='1000000001'  index_to_float(v)='-458.422'
float_to_index(index_to_float(v))='1000000001' v='2000000000'
index_to_float(v)='-6.09141e-34'  float_to_index(index_to_float(v))='2000000000'
v='2000000001'  index_to_float(v)='-6.09141e-34'
float_to_index(index_to_float(v))='2000000001' v='3000000000'
index_to_float(v)='4.85143e-08'  float_to_index(index_to_float(v))='1278190081'
v='3000000001'  index_to_float(v)='4.85143e-08'
float_to_index(index_to_float(v))='1278190080' v='4026531838'
index_to_float(v)='3.16913e+29'  float_to_index(index_to_float(v))='251658243'
v='4026531839'  index_to_float(v)='3.16913e+29'
float_to_index(index_to_float(v))='251658242'


*/
  auto vs{std::vector<uint32_t>({0, 1, 12, 1000, 10000, 100000, 1000000000,
                                 1000000001, 2000000000, 2000000001, 3000000000,
                                 3000000001, 4026531839 - 1, 4026531839})};
  for (const auto &v : vs) {
    std::cout << ""
              << " v='" << v << "' "
              << " index_to_float(v)='" << index_to_float(v) << "' "
              << " float_to_index(index_to_float(v))='"
              << float_to_index(index_to_float(v)) << "' " << std::endl;
  }
  return 0;
}
