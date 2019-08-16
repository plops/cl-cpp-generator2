#include <wmma.h>
// independent dot products;
// inefficient due to large working sets to hold parts of A and B;
for (i = 0; i < M; (i) += (1)) {
  for (j = 0; j < N; (j) += (1)) {
    for (k = 0; k < K; (k) += (1)) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
// accumulate outer products;
// load elements of A and B exactly once;
for (k = 0; k < K; (k) += (1)) {
  for (i = 0; i < M; (i) += (1)) {
    for (j = 0; j < N; (j) += (1)) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
// partition into Mtile-by-Ntile independent matrix products;
for (mb = 0; mb < M; (mb) += (Mtile)) {
  for (nb = 0; nb < N; (nb) += (Ntile)) {
    for (kb = 0; kb < K; (kb) += (Ktile)) {
      for (k = 0; k < Ktile; (k) += (1)) {
        for (i = 0; i < Mtile; (i) += (1)) {
          for (j = 0; j < Ntile; (j) += (1)) {
            auto row = ((mb) + (i));
            auto col = ((nb) + (j));
            (C[row][col]) +=
                (((A[row][((kb) + (k))]) * (B[((kb) + (k))][col])));
          }
        }
      }
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