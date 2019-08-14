#include <wmma.h>
__device__ void tensor_op_16_16_16(float *d, half *a, half *b, float *c) {
  wmma::fragment<matrix_a, ...> Amat = ;
  wmma::fragment<matrix_b, ...> Bmat = ;
  auto Cmat = ;
  wmma::load_matrix_sync(Amat, a, 16);
  wmma::load_matrix_sync(Bmat, b, 16);
  wmma::fill_fragment(Cmat, .0s);
  wmma::mma_sync(Cmat, Amat, Bmat, Cmat);
  wmma::store_matrix_sync(d, Cmat, 16, wmma::row_major);
}