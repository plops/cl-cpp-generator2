#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
;
class Points {
  float *m_x, *m_y;

public:
  __host__ __device__ Points();
  __host__ __device__ Points(float *x, float *y);
  __host__ __device__ __forceinline__ float2 get_point(int idx) const;
  __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p);
  __host__ __device__ __forceinline__ void set(float *x, float *y);
};
class Bounding_box {
  float2 m_p_min, m_p_max;

public:
  __host__ __device__ Bounding_box();
  __host__ __device__ void compute_center(float2 &center) const;
  __host__ __device__ __forceinline__ const float2 &get_min() const;
  __host__ __device__ __forceinline__ const float2 &get_max() const;
  __host__ __device__ bool contains(const float2 &p) const;
  __host__ __device__ void set(float xmin, float ymin, float xmax, float ymax);
};
class Quadtree_node {
  int m_id, m_begin, m_end;
  Bounding_box m_bounding_box;

public:
  __host__ __device__ Quadtree_node();
  __host__ __device__ int id() const;
  __host__ __device__ void set_id(int new_id);
  __host__ __device__ __forceinline__ const Bounding_box bounding_box() const;
  __host__ __device__ __forceinline__ void
  set_bounding_box(float xmin, float ymin, float xmax, float ymax);
  __host__ __device__ __forceinline__ int num_points() const;
  __host__ __device__ __forceinline__ int points_begin() const;
  __host__ __device__ __forceinline__ int points_end() const;
  __host__ __device__ __forceinline__ void set_range(int begin, int end);
};
class Parameters {
public:
  int m_point_selector, m_num_nodes_at_this_level, m_depth;
  const int m_max_depth, m_min_points_per_node;
  __host__ __device__ Parameters(int max_depth, int min_points_per_node);
  __host__ __device__ Parameters(const Parameters &params, bool flag);
};
class Random_generator {
public:
  int count;
  __host__ __device__ Random_generator();
  __host__ __device__ unsigned int hash(unsigned int a);
  __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()();
};
