
#include "utils.h"

#include "globals.h"

// implementation
;
#include <vis_00_base.hpp>

using namespace std::chrono_literals;
using namespace thrust;

State state = {};
int jacobi_solver(float *data, int M, int N, float max_diff) {
  // The Jacobi method consists of approximating the square plate with a
  // two-dimensional grid of points. A two-dimensional array is used to
  // represent the temperature at each of these points. Each iteration updates
  // the elements of the array from the values computed at the previous step,
  // using the following update scheme:
  // T^{n+1}_ {i, j} = 0.25 * (T^n_ {i-1, j} + T^n_{i+1, j} + T^n_{i, j-1}
  // +T^n_{i, j+1}) This is repeated until convergence is reached: when the
  // values obtained at the end of two subsequent iterations do not differ
  // significantly. You will notice that the update is impossible to perform at
  // the boundary grid points, i.e., the edges of the plate. There are many ways
  // to treat boundaries, but in this simple example, we will simply assume that
  // their temperature doesn t change (fixed or Dirichlet boundary condition).
  // c++: 5min49s
  // https://on-demand.gputechconf.com/supercomputing/2019/video/sc1936-gpu-programming-with-standard-c++17/

  // python:
  // https://developer.nvidia.com/blog/accelerating-python-on-gpus-with-nvc-and-cython/
  //  https://github.com/shwina/stdpar-cython/blob/main/jacobi.ipynb
  ;
  auto temp = std::make_unique<float[]>(((M) * (N)));
  std::copy(std::execution::par, data, ((data) + (((M) * (N)))), temp.get());
  auto iterations = 0;
  auto keep_going = true;
  auto from = data;
  auto to = temp.get();
  while (keep_going) {
    (iterations)++;
    std::for_each(std::execution::par, counting_iterator<int>(((N) + (1))),
                  counting_iterator<int>(((((((M) - (1))) * (N))) - (1))),
                  [=](int i) {
                    if ((((0) != (i % N)) && ((((N) - (1))) != (i % N)))) {
                      to[i] = (((0.250f)) *
                               (((from[((i) - (N))]) + (from[((i) + (N))]) +
                                 (from[((i) - (1))]) + (from[((i) + (1))]))));
                    }
                  });
    keep_going = std::any_of(
        std::execution::par, counting_iterator<int>(((N) + (1))),
        counting_iterator<int>(((((((M) - (1))) * (N))) - (1))),
        [=](int i) { return (max_diff) < (std::fabs(((to[i]) - (from[i])))); });
    std::swap(from, to);
  }
  if ((to) == (data)) {
    std::copy(std::execution::par, temp.get(), ((temp.get()) + (((M) * (N)))),
              data);
  }
  return iterations;
}
int main(int argc, char **argv) {
  state._main_version = "2ba0ec86059b381c357f34d4b377abe6a1a2f218";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/58_stdpar/source/";
  state._code_generation_time = "14:23:13 of Monday, 2020-12-28 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start main") << (" ") << (std::setw(8))
                << (" state._main_version='") << (state._main_version) << ("'")
                << (std::setw(8)) << (" state._code_repository='")
                << (state._code_repository) << ("'") << (std::setw(8))
                << (" state._code_generation_time='")
                << (state._code_generation_time) << ("'") << (std::endl)
                << (std::flush);
  }
  const int M = 128;
  const int N = 128;
  auto data = std::vector<float>();
  data.resize(((M) * (N)));
  std::fill(data.begin(), data.end(), 0);
  for (auto i = 0; (i) < (N); (i) += (1)) {
    data[((i) + (((M) * (0))))] = std::sin(
        ((static_cast<float>(M_PI)) * (((static_cast<float>(i)) / (N)))));
  }
  jacobi_solver(data.data(), M, N, (1.00e-5f));
  return 0;
}
