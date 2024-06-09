
# Summary of the Blog Post in [1]

**The Problem:**

* Generating every single floating-point number in order, with a bijective mapping between integers (indices) and floats.
* Existing methods using `nextafter()` have limitations: no int-to-float mapping, ±0 treated as one value, and NaNs not generated.

**The Solution:**

* **Exploiting IEEE754 format:** Floats are almost ordered when their bit patterns are interpreted as uint32.
* **Handling positive numbers:** Sorting uint32 representation sorts positive floats due to exponent and mantissa order.
* **Handling negative numbers:** Complementing the bits of negative floats reverses their order, aligning with uint32 sorting.
* **Function `to_float(n)`:** Takes a uint32 `n` and converts it to its corresponding float based on the above logic.

**Key Features:**

* **Bijective mapping:** Every integer in the range maps to a unique float.
* **Ordered output:** Floats are generated in ascending order from -Inf to +Inf.
* **NaN handling:** All NaN values are grouped after +Inf.
* **Performance:** Fast enough to iterate through all ~4 billion floats on modern hardware.

**Code Example:**

```c++
float to_float(uint32_t n){
    n += ((1<<23)-1);
    float f;
    if(n & (1<<31))
        n ^= (1<<31);
    else
        n=~n;
    memcpy(&f, &n, 4);
    return f;   
}
```

**Conclusion:**

This blog post presents an efficient and elegant solution for generating and indexing every possible single-precision floating-point number by leveraging the inherent order in the IEEE754 representation.


# Inverse Transform


```c++
uint32_t float_to_index(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));

  if (u & (1u << 31)) {
    u ^= (1u << 31); 
  } else {
    u = ~u;
  }

  return u - ((1u << 23) - 1);
}
```

**Explanation:**

1. **Bitwise Copy:** We first copy the bit representation of the float `f` into a `uint32_t` variable `u`. This is done using `memcpy`.

2. **Reverse Transformation:** We then apply the reverse of the transformation used in `to_float`.
   * If the sign bit (most significant bit) is 1 (negative number), we flip the sign bit.
   * If the sign bit is 0 (positive number or NaN), we complement all the bits.

3. **Subtract Offset:** Finally, we subtract the offset `((1u << 23) - 1)` which was added in the `to_float` function to get the original index.

**Note:**

* This function assumes the input float `f` was generated using the `to_float` function presented in the article.
*  It does not handle the special case of `-0` and `+0` being mapped to the same index. If you need to distinguish between them, you'll need to add an additional check for the zero value. 



# Reference

[1] https://deathandthepenguinblog.wordpress.com/2022/12/31/every-single-float-in-order-with-random-access/#comments
