

kernel void add_kernel(global const float *A, global const float *B,
                       global float restrict *C) {
  const uint n{get_global_id(0)};
  ((C)[(n)]) = (((A)[(n)]) + ((B)[(n)]));
}
