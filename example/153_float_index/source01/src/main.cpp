#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
// This is a mapping between floating point numbers and integer indices that
// keep order
/**

1. **Initialization and Offset:**
   - It takes an unsigned 32-bit integer `n` as input.
   - An offset of `(1U << 23) - 1` (which is 8388607) is added to `n`. This
offset is important for aligning the integer representation with the bias used
in the IEEE 754 standard for floating-point numbers.

2. **Sign Bit Manipulation:**
   - The code checks the most significant bit (sign bit) of `n`.
   - If the sign bit is set (`n & (1U << 31) != 0u`):
     - It toggles the sign bit (`n ^= (1U << 31)`). This maps positive integers
to negative floats and vice versa.
   - If the sign bit is not set:
     - It takes the bitwise complement of `n` (`n = ~n`). This ensures the
correct mapping of negative floats.

3. **Conversion to Float:**
   -  A `float` variable `f` is declared.
   -  The bit pattern of `n` is copied directly into `f` using `memcpy`. This is
the key step where the integer bit representation is reinterpreted as a
floating-point representation.

4. **Return Value:** The function returns the float `f`.

*/

float index_to_float(uint32_t n) {
  n += ((1U << 23) - 1);
  if ((n & (1U << 31)) != 0u) {
    n = n ^ (1U << 31);
  } else {
    n = ~n;
  }
  float f;
  memcpy(&f, &n, 4);
  return f;
}

/**
1. **Bitwise Copy:** We first copy the bit representation of the float `f` into
a `uint32_t` variable `u`. This is done using `memcpy`.

2. **Reverse Transformation:** We then apply the reverse of the transformation
used in `to_float`.
   * If the sign bit (most significant bit) is 1 (negative number), we flip the
sign bit.
   * If the sign bit is 0 (positive number or NaN), we complement all the bits.

3. **Subtract Offset:** Finally, we subtract the offset `((1u << 23) - 1)` which
was added in the `to_float` function to get the original index.

*/

uint32_t float_to_index(float f) {
  uint32_t n;
  memcpy(&n, &f, sizeof(n));
  if ((n & (1U << 31)) != 0u) {
    n = ~(n ^ (1U << 31));
  } else {
    n = ~n;
  }
  // Ensure the subtraction is done as unsigned
  return n - ((1U << 31) + ((1U << 23) - 1U));
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
// from here the inverse function not working v='3000000001'
index_to_float(v)='4.85143e-08'  float_to_index(index_to_float(v))='1278190080'
v='4026531838'  index_to_float(v)='3.16913e+29'
float_to_index(index_to_float(v))='251658243' v='4026531839'
index_to_float(v)='3.16913e+29'  float_to_index(index_to_float(v))='251658242'


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
