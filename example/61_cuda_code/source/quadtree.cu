#include "quadtree.h"
__host__ __device__ Points::Points() : m_x{NULL}, m_y{NULL} {}
__host__ __device__ Points::Points(float *x, float *y) : m_x{x}, m_y{y} {}
__host__ __device__ __forceinline__ float2 Points::get_point(int idx) const {
  return make_float2((m_x)[(idx)], (m_y)[(idx)]);
}
__host__ __device__ __forceinline__ void Points::set_point(int idx,
                                                           const float2 &p) {
  ((m_x)[(idx)]) = (p.x);
  ((m_y)[(idx)]) = (p.y);
}
__host__ __device__ __forceinline__ void Points::set(float *x, float *y) {
  (m_x) = (x);
  (m_y) = (y);
}
__host__ __device__ Bounding_box::Bounding_box() {
  (m_p_min) = (make_float2(0.F, 0.F));
  (m_p_max) = (make_float2(1.0F, 1.0F));
}
__host__ __device__ void Bounding_box::compute_center(float2 &center) const {
  (center.x) = ((0.50F) * ((m_p_min.x) + (m_p_max.x)));
  (center.y) = ((0.50F) * ((m_p_min.y) + (m_p_max.y)));
}
__host__ __device__ __forceinline__ const float2 &
Bounding_box::get_min() const {
  return m_p_min;
}
__host__ __device__ __forceinline__ const float2 &
Bounding_box::get_max() const {
  return m_p_max;
}
__host__ __device__ bool Bounding_box::contains(const float2 &p) const {
  return ((m_p_min.x) <= (p.x)) & ((p.x) <= (m_p_max.x)) &
         ((m_p_min.y) <= (p.y)) & ((p.y) <= (m_p_max.y));
}
__host__ __device__ void Bounding_box::set(float xmin, float ymin, float xmax,
                                           float ymax) {
  (m_p_min.x) = (xmin);
  (m_p_min.y) = (ymin);
  (m_p_max.x) = (xmax);
  (m_p_max.y) = (ymax);
}
__host__ __device__ Quadtree_node::Quadtree_node()
    : m_id{0}, m_begin{0}, m_end{0} {}
__host__ __device__ int Quadtree_node::id() const { return m_id; }
__host__ __device__ void Quadtree_node::set_id(int new_id) {
  (m_id) = (new_id);
}
__host__ __device__ __forceinline__ const Bounding_box
Quadtree_node::bounding_box() const {
  return m_bounding_box;
}
__host__ __device__ __forceinline__ void
Quadtree_node::set_bounding_box(float xmin, float ymin, float xmax,
                                float ymax) {
  m_bounding_box.set(xmin, ymin, xmax, ymax);
}
__host__ __device__ __forceinline__ int Quadtree_node::num_points() const {
  return (m_end) - (m_begin);
}
__host__ __device__ __forceinline__ int Quadtree_node::points_begin() const {
  return m_begin;
}
__host__ __device__ __forceinline__ int Quadtree_node::points_end() const {
  return m_end;
}
__host__ __device__ __forceinline__ void Quadtree_node::set_range(int begin,
                                                                  int end) {
  (m_begin) = (begin);
  (m_end) = (end);
}
__host__ __device__ Parameters::Parameters(int max_depth,
                                           int min_points_per_node)
    : m_point_selector{0}, m_num_nodes_at_this_level{1}, m_depth{0},
      m_max_depth{max_depth}, m_min_points_per_node{min_points_per_node} {}
__host__ __device__ Parameters::Parameters(const Parameters &params, bool flag)
    : m_point_selector{((params.m_point_selector) + (1)) % (2)},
      m_num_nodes_at_this_level{(4) * (params.m_num_nodes_at_this_level)},
      m_depth{(1) + (params.m_depth)}, m_max_depth{params.m_max_depth},
      m_min_points_per_node{params.m_min_points_per_node} {}
__host__ __device__ Random_generator::Random_generator() : count{0} {}
__host__ __device__ unsigned int Random_generator::hash(unsigned int a) {
  (a) = (((a) + (0b01111110110101010101110100010110)) + ((a) << (12)));
  (a) = (((a) ^ (0b11000111011000011100001000111100)) ^ ((a) >> (19)));
  (a) = (((a) + (0b00010110010101100110011110110001)) + ((a) << (5)));
  (a) = (((a) + (0b11010011101000100110010001101100)) ^ ((a) << (9)));
  (a) = (((a) + (0b11111101011100000100011011000101)) + ((a) << (3)));
  (a) = (((a) ^ (0b10110101010110100100111100001001)) ^ ((a) >> (16)));
  return a;
}
__host__ __device__ __forceinline__ thrust::tuple<float, float>
Random_generator::operator()() {
#ifdef __CUDA_ARCH__
  auto seed{hash((count) + (threadIdx.x) + ((blockDim.x) * (blockIdx.x)))};
  (count) += ((blockDim.x) * (gridDim.x));
#else
  auto seed{hash(0)};
#endif
  thrust::default_random_engine rng(seed);
  thrust::random::uniform_real_distribution<float> distrib;
  return thrust::make_tuple(distrib(rng), distrib(rng));
}
namespace cg = cooperative_groups;
#include <helper_cuda.h>

template <int NUM_THREADS_PER_BLOCK>
__global__ void build_quadtree_kernel(Quadtree_node *nodes, Points *points,
                                      Parameters params) {
  auto cta{cg::this_thread_block()};
  auto NUM_WARPS_PER_BLOCK{(NUM_THREADS_PER_BLOCK) / (warpSize)};
  auto warp_id{(threadIdx.x) / (warpSize)};
  auto lane_id{(threadIdx.x) % (warpSize)};
  auto lane_mask_lt{((lane_id)) - (1)};
  extern __shared__ int smem[];
  volatile int *s_num_pts[4];
  auto &node{(nodes)[(blockIdx.x)]};
  auto num_points{node.num_points()};
#pragma unroll
  for (auto i = 0; (i) < (4); (i) += (1)) {
    ((s_num_pts)[(i)]) =
        ((volatile int *)&((smem)[((i) * (NUM_WARPS_PER_BLOCK))]));
  }
  // stop recursion here, points[0] contains all points

  float2 center;
  int range_begin;
  int range_end;
  int warp_counts[4]{{0, 0, 0, 0}};
  if (((params.m_max_depth) <= (params.m_depth)) |
      ((num_points) <= (params.m_min_points_per_node))) {
    if ((1) == (params.m_point_selector)) {
      auto it{node.points_begin()};
      auto end{node.points_end()};
#pragma unroll
      for ((it) += (threadIdx.x); (it) < (end);
           (it) += (NUM_THREADS_PER_BLOCK)) {
        if ((it) < (end)) {
          (points)[(0)].set_point(it, (points)[(1)].get_point(it));
        }
      }
    }
    return;
  }
  // find number of points in each warp, and points to move to each quadrant

  const Bounding_box &bbox{node.bounding_box()};
  bbox.compute_center(center);
  auto num_points_per_warp{
      max(warpSize, ((num_points) + (NUM_WARPS_PER_BLOCK) + (-1)) /
                        (NUM_WARPS_PER_BLOCK))};
  (range_begin) = ((node.points_begin()) + ((warp_id) * (num_points_per_warp)));
  (range_end) = (min((range_begin) + (num_points_per_warp), node.points_end()));
  // count points in each child

  auto &in_points{(points)[(params.m_point_selector)]};
  auto tile32{cg::tiled_partition<32>(cta)};
#pragma unroll
  for ((int range_it) = ((range_begin) + (tile32.thread_rank()));
       tile32.any((range_it) < (range_end)); (range_it) += (warpSize)) {
    auto is_active{(range_it) < (range_end)};
    auto p{(is_active) ? (in_points.get_point(range_it))
                       : (make_float2(0.F, 0.F))};
    {
      auto num_pts{__popc(tile32.ballot(
          (is_active) & (((p.x) < (center.x)) & ((center.y) <= (p.y)))))};
      ((warp_counts)[(0)]) += (tile32.shfl(num_pts, 0));
    }
    {
      auto num_pts{__popc(tile32.ballot(
          (is_active) & (((center.x) <= (p.x)) & ((center.y) <= (p.y)))))};
      ((warp_counts)[(1)]) += (tile32.shfl(num_pts, 0));
    }
    {
      auto num_pts{__popc(tile32.ballot(
          (is_active) & (((p.x) < (center.x)) & ((p.y) < (center.y)))))};
      ((warp_counts)[(2)]) += (tile32.shfl(num_pts, 0));
    }
    {
      auto num_pts{__popc(tile32.ballot(
          (is_active) & (((center.x) <= (p.x)) & ((p.y) < (center.y)))))};
      ((warp_counts)[(3)]) += (tile32.shfl(num_pts, 0));
    }
  }
  if ((0) == (tile32.thread_rank())) {
    (((s_num_pts)[(0)])[(warp_id)]) = ((warp_counts)[(0)]);
    (((s_num_pts)[(1)])[(warp_id)]) = ((warp_counts)[(1)]);
    (((s_num_pts)[(2)])[(warp_id)]) = ((warp_counts)[(2)]);
    (((s_num_pts)[(3)])[(warp_id)]) = ((warp_counts)[(3)]);
  }
  cg::sync(cta);
  // scan warps' results

  if ((warp_id) < (4)) {
    auto num_pts{((tile32.thread_rank()) < (NUM_WARPS_PER_BLOCK))
                     ? (((s_num_pts)[(warp_id)])[(tile32.thread_rank())])
                     : (0)};
#pragma unroll
    for ((int offset) = (1); (offset) < (NUM_WARPS_PER_BLOCK);
         (offset) *= (2)) {
      auto n{tile32.shfl_up(num_pts, offset)};
      if ((offset) <= (tile32.thread_rank())) {
        (num_pts) += (n);
      }
    }
    if ((tile32.thread_rank()) < (NUM_WARPS_PER_BLOCK)) {
      (((s_num_pts)[(warp_id)])[(tile32.thread_rank())]) = (num_pts);
    }
  }
  cg::sync(cta);
  // global offset

  if ((0) == (warp_id)) {
    auto sum{(s_num_pts)[(0)][((NUM_WARPS_PER_BLOCK) - (1))]};
    for ((int row) = (1); (row) < (4); (row)++) {
      auto tmp{(s_num_pts)[(row)][((NUM_WARPS_PER_BLOCK) - (1))]};
      cg::sync(tile32);
      if ((tile32.thread_rank()) < (NUM_WARPS_PER_BLOCK)) {
        ((s_num_pts)[(row)][(tile32.thread_rank())]) += (sum);
      }
      cg::sync(tile32);
      (sum) += (tmp);
    }
  }
  cg::sync(cta);
  // scan exclusive

  auto val{0};
  if ((threadIdx.x) < ((4) * (NUM_WARPS_PER_BLOCK))) {
    (val) = (((0) == (threadIdx.x)) ? (0) : ((smem)[((threadIdx.x) - (1))]));
    (val) += (node.points_begin());
  }
  cg::sync(cta);
  // move points

  if (!(((params.m_max_depth) <= (params.m_depth)) |
        ((num_points) <= (params.m_min_points_per_node)))) {
    auto &out_points{(points)[(((params.m_point_selector) + (1)) % (2))]};
    ((warp_counts)[(0)]) = ((s_num_pts)[(0)][(warp_id)]);
    ((warp_counts)[(1)]) = ((s_num_pts)[(1)][(warp_id)]);
    ((warp_counts)[(2)]) = ((s_num_pts)[(2)][(warp_id)]);
    ((warp_counts)[(3)]) = ((s_num_pts)[(3)][(warp_id)]);
    auto &in_points{(points)[(params.m_point_selector)]};
    // reorder points

    for ((int range_it) = ((range_begin) + (tile32.thread_rank()));
         tile32.any((range_it) < (range_end)); (range_it) += (warpSize)) {
      auto is_active{(range_it) < (range_end)};
      auto p{(is_active) ? (in_points.get_point(range_it))
                         : (make_float2(0.F, 0.F))};
      {
        auto pred{(is_active) & (((p.x) < (center.x)) & ((center.y) <= (p.y)))};
        auto vote{tile32.ballot(pred)};
        auto dest{((warp_counts)[(0)]) + (__popc((vote) && (lane_mask_lt)))};
        if (pred) {
          out_points.set_point(dest, p);
        }
        ((warp_counts)[(0)]) += (tile32.shfl(__popc(vote), 0));
      }
      {
        auto pred{(is_active) &
                  (((center.x) <= (p.x)) & ((center.y) <= (p.y)))};
        auto vote{tile32.ballot(pred)};
        auto dest{((warp_counts)[(1)]) + (__popc((vote) && (lane_mask_lt)))};
        if (pred) {
          out_points.set_point(dest, p);
        }
        ((warp_counts)[(1)]) += (tile32.shfl(__popc(vote), 0));
      }
      {
        auto pred{(is_active) & (((p.x) < (center.x)) & ((p.y) < (center.y)))};
        auto vote{tile32.ballot(pred)};
        auto dest{((warp_counts)[(2)]) + (__popc((vote) && (lane_mask_lt)))};
        if (pred) {
          out_points.set_point(dest, p);
        }
        ((warp_counts)[(2)]) += (tile32.shfl(__popc(vote), 0));
      }
      {
        auto pred{(is_active) & (((center.x) <= (p.x)) & ((p.y) < (center.y)))};
        auto vote{tile32.ballot(pred)};
        auto dest{((warp_counts)[(3)]) + (__popc((vote) && (lane_mask_lt)))};
        if (pred) {
          out_points.set_point(dest, p);
        }
        ((warp_counts)[(3)]) += (tile32.shfl(__popc(vote), 0));
      }
    }
  }
  cg::sync(cta);
  if ((0) == (tile32.thread_rank())) {
    ((s_num_pts)[(0)][(warp_id)]) = ((warp_counts)[(0)]);
    ((s_num_pts)[(1)][(warp_id)]) = ((warp_counts)[(1)]);
    ((s_num_pts)[(2)][(warp_id)]) = ((warp_counts)[(2)]);
    ((s_num_pts)[(3)][(warp_id)]) = ((warp_counts)[(3)]);
  }
  cg::sync(cta);
  // launch new blocks

  if (!(((params.m_max_depth) <= (params.m_depth)) |
        ((num_points) <= (params.m_min_points_per_node)))) {
    if (((NUM_THREADS_PER_BLOCK) - (1)) == (threadIdx.x)) {
      auto *children{&((nodes)[((params.m_num_nodes_at_this_level) -
                                ((node.id()) && (~3)))])};
      auto child_offset{(4) * (node.id())};
      (children)[((child_offset) + (0))].set_id((0) + ((4) * (node.id())));
      (children)[((child_offset) + (1))].set_id((1) + ((4) * (node.id())));
      (children)[((child_offset) + (2))].set_id((2) + ((4) * (node.id())));
      (children)[((child_offset) + (3))].set_id((3) + ((4) * (node.id())));
      auto &bbox{node.bounding_box()};
      auto &p_min{bbox.get_min()};
      auto &p_max{bbox.get_max()};
      // set bboxes of the children

      (children)[((child_offset) + (0))].set_bounding_box(p_min.x, center.y,
                                                          center.x, p_max.y);
      (children)[((child_offset) + (1))].set_bounding_box(center.x, center.y,
                                                          p_max.x, p_max.y);
      (children)[((child_offset) + (2))].set_bounding_box(p_min.x, p_min.y,
                                                          center.x, center.y);
      (children)[((child_offset) + (3))].set_bounding_box(center.x, p_min.y,
                                                          p_max.x, center.y);
      // set ranges of children

      (children)[((0) + (child_offset))].set_range(node.points_begin(),
                                                   (s_num_pts)[(0)][(warp_id)]);
      (children)[((1) + (child_offset))].set_range((s_num_pts)[(0)][(warp_id)],
                                                   (s_num_pts)[(1)][(warp_id)]);
      (children)[((2) + (child_offset))].set_range((s_num_pts)[(1)][(warp_id)],
                                                   (s_num_pts)[(2)][(warp_id)]);
      (children)[((3) + (child_offset))].set_range((s_num_pts)[(2)][(warp_id)],
                                                   (s_num_pts)[(3)][(warp_id)]);
      // launch children

      build_quadtree_kernel<NUM_THREADS_PER_BLOCK>
          <<<4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
              &((children)[(child_offset)]), points, Parameters(params, true));
    }
  }
}

bool cdpQuadTree(int warp_size) {
  auto num_points{1024};
  auto max_depth{8};
  auto min_points_per_node{16};
  thrust::device_vector<float> x_d0(num_points);
  thrust::device_vector<float> x_d1(num_points);
  thrust::device_vector<float> y_d0(num_points);
  thrust::device_vector<float> y_d1(num_points);
  Random_generator rnd;
  thrust::generate(
      thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(), y_d0.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(x_d0.end(), y_d0.end())),
      rnd);
  // host structures to analyze device structures

  Points points_init[2];
  (points_init)[(0)].set(thrust::raw_pointer_cast(&((x_d0)[(0)])),
                         thrust::raw_pointer_cast(&((y_d0)[(0)])));
  (points_init)[(1)].set(thrust::raw_pointer_cast(&((x_d1)[(0)])),
                         thrust::raw_pointer_cast(&((y_d1)[(0)])));
  // allocate memory to store points

  Points *points;
  checkCudaErrors(cudaMalloc((void **)&(points), (2) * (sizeof(Points))));
  checkCudaErrors(cudaMemcpy(points, points_init, (2) * (sizeof(Points)),
                             cudaMemcpyHostToDevice));
  auto max_nodes{0};
  auto num_nodes_at_level{1};
  for (auto i = 0; (i) < (max_depth); (i) += (1)) {
    (max_nodes) += (num_nodes_at_level);
    (num_nodes_at_level) *= (4);
  }
  Quadtree_node root, *nodes;
  root.set_range(0, num_points);
  checkCudaErrors(
      cudaMalloc((void **)&(nodes), (max_nodes) * (sizeof(Quadtree_node))));
  checkCudaErrors(
      cudaMemcpy(nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice));
  // recursion limit to max_depth

  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
  // build quadtree

  Parameters params(max_depth, min_points_per_node);
  const int NUM_THREADS_PER_BLOCK{128};
  const int NUM_WARPS_PER_BLOCK{(NUM_THREADS_PER_BLOCK) / (warp_size)};
  const size_t smem_size{(4) * (NUM_WARPS_PER_BLOCK) * (sizeof(int))};
  build_quadtree_kernel<NUM_THREADS_PER_BLOCK>
      <<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(nodes, points, params);
  checkCudaErrors(cudaGetLastError());
  return true;
}

int main(int argc, char **argv) {
  auto cuda_device{findCudaDevice(argc, (const char **)argv)};
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
  auto ok{cdpQuadTree(deviceProps.warpSize)};
  return (ok) ? (EXIT_SUCCESS) : (EXIT_FAILURE);
}
