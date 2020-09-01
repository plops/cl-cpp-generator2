
#include "utils.h"

#include "globals.h"

extern State state;
#include "mkl.h"
#include <chrono>
#include <iostream>
#include <thread>
void init_matrix(double *a, int m, int n) {
  for (auto i = 0; (i) < (((m) * (n))); (i) += (1)) {
    a[i] = static_cast<double>(rand() % 100);
  }
}
std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}
//  https://github.com/CoffeeBeforeArch/hp_cpp/blob/master/libraries/mkl/dgemm.cpp
int main(int argc, char **argv) {
  double *A, *B, *C;
  auto m = 2000;
  auto k = 200;
  auto n = 1000;
  auto alpha = (1.0);
  auto beta = (0.);
  A = static_cast<double *>(mkl_malloc(((m) * (k) * (sizeof(double))), 64));
  init_matrix(A, m, k);
  B = static_cast<double *>(mkl_malloc(((k) * (n) * (sizeof(double))), 64));
  init_matrix(B, k, n);
  C = static_cast<double *>(mkl_malloc(((m) * (n) * (sizeof(double))), 64));
  init_matrix(C, m, n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k,
              B, n, beta, C, n);
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
}