#include <wmma.h>
// independent dot products;
// inefficient due to large working sets to hold parts of A and B;
for (i = 0;; i < M; (i)++) {
  for (j = 0;; j < N; (j)++) {
    for (k = 0;; k < K; (k)++) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
// accumulate outer products;
// load elements of A and B exactly once;
for (k = 0;; k < K; (k)++) {
  for (i = 0;; i < M; (i)++) {
    for (j = 0;; j < N; (j)++) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
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