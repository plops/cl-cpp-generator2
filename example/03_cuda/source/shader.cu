#include <wmma.h>
// independent dot products
// inefficient due to large working sets to hold parts of A and B
for (i = 0; i < M; (i) += (1)) {
  for (j = 0; j < N; (j) += (1)) {
    for (k = 0; k < K; (k) += (1)) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
// accumulate outer products
// load elements of A and B exactly once
for (k = 0; k < K; (k) += (1)) {
  for (i = 0; i < M; (i) += (1)) {
    for (j = 0; j < N; (j) += (1)) {
      (C[i][j]) += (((A[i][k]) * (B[k][j])));
    }
  }
}
// partition into Mtile-by-Ntile independent matrix products
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
// each warp computes independent matrix product
for (mb = 0; mb < M; (mb) += (Mtile)) {
  for (nb = 0; nb < N; (nb) += (Ntile)) {
    for (kb = 0; kb < K; (kb) += (Ktile)) {
      // load A and B tiles into shared memory
      for (m = 0; m < Mtile; (m) += (warp_m)) {
        for (n = 0; n < Ntile; (n) += (warp_n)) {
          for (k = 0; k < Ktile; (k) += (warp_k)) {
            // compute warp_m by warp_n by warp_k GEMM
          }
        }
      }
    }
  }
}
// accumulated matrix product in warps
for (mb = 0; mb < M; (mb) += (Mtile)) {
  for (nb = 0; nb < N; (nb) += (Ntile)) {
    for (kb = 0; kb < K; (kb) += (Ktile)) {
      // load A and B tiles into shared memory
      for (m = 0; m < Mtile; (m) += (warp_m)) {
        for (n = 0; n < Ntile; (n) += (warp_n)) {
          for (k = 0; k < Ktile; (k) += (warp_k)) {
            // load A and B tile from SMEM into registers
            for (tm = 0; tm < warp_m; (tm) += (thread_m)) {
              for (tn = 0; tn < warp_n; (tn) += (thread_n)) {
                for (tk = 0; tk < warp_k; (tk) += (thread_k)) {
                  // compute thread_m by thread_n by thread_k GEMM
                }
              }
            }
          }
        }
      }
    }
  }
}
// threads compute accumulated matrix product
// A,B and C held in registers
// O(M*N) computations on O(M+N) elements
// opportunity for data reuse
for (mb = 0; mb < M; (mb) += (Mtile)) {
  for (nb = 0; nb < N; (nb) += (Ntile)) {
    for (kb = 0; kb < K; (kb) += (Ktile)) {
      // load A and B tiles into shared memory
      for (m = 0; m < Mtile; (m) += (warp_m)) {
        for (n = 0; n < Ntile; (n) += (warp_n)) {
          for (k = 0; k < Ktile; (k) += (warp_k)) {
            // load A and B tile from SMEM into registers
            for (tm = 0; tm < warp_m; (tm) += (thread_m)) {
              for (tn = 0; tn < warp_n; (tn) += (thread_n)) {
                for (tk = 0; tk < warp_k; (tk) += (thread_k)) {
                  for (m = 0; m < thread_m; (m) += (1)) {
                    for (n = 0; n < thread_n; (n) += (1)) {
                      for (k = 0; k < thread_k; (k) += (1)) {
                        // FMA instructions
                        (C[m][n]) += (((A[m][k]) * (B[n][k])));
                      }
                    }
                  }
                }
              }
            }
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