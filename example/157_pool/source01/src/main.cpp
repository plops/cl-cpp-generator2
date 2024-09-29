#include <algorithm>
#include <array>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>
using namespace std;
// cppcon 2017 Pablo Halpern Allocators: The Good Parts timestamp 30:46
// Klaus Iglberger: C++ Software Design pp. 142
class test_resource : public pmr::memory_resource {
public:
  test_resource(pmr::memory_resource *upstream) : _upstream{upstream} {
    auto upstream_ptr{reinterpret_cast<uint64_t>(_upstream)};
    std::cout << std::format("(make_test_resource :upstream_ptr '{}')\n",
                             upstream_ptr);
  }
  ~test_resource() {}
  pmr::memory_resource *upstream() const { return _upstream; }
  size_t bytes_allocated() const { return _bytes_allocated; }
  size_t bytes_deallocated() const { return 0; }
  size_t bytes_outstanding() const { return _bytes_outstanding; }
  size_t bytes_highwater() const { return _bytes_highwater; }
  size_t blocks_outstanding() const { return 0; }
  // We can't throw in the destructor that is why we need the following three
  // functions
  size_t leaked_bytes() { return _s_leaked_bytes; }
  size_t leaked_blocks() { return _s_leaked_blocks; }
  void clear_leaked() {
    _s_leaked_bytes = 0;
    _s_leaked_blocks = 0;
  }

protected:
  void *do_allocate(size_t bytes, size_t alignment) override {
    std::cout << std::format("(do_allocate :bytes '{}' :alignment '{}')\n",
                             bytes, alignment);
    auto ret{_upstream->allocate(bytes, alignment)};
    _blocks.push_back(allocation_rec{ret, bytes, alignment});
    _bytes_allocated += bytes;
    _bytes_outstanding += bytes;
    if (_bytes_highwater < _bytes_outstanding) {
      _bytes_highwater = _bytes_outstanding;
    }
    return ret;
  }
  void do_deallocate(void *p, size_t bytes, size_t alignment) override {
    std::cout << std::format(
        "(do_deallocate :p '{}' :bytes '{}' :alignment '{}')\n", p, bytes,
        alignment);
    auto i{std::find_if(_blocks.begin(), _blocks.end(),
                        [p](allocation_rec &r) { return r._ptr == p; })};
    if (i == _blocks.end()) {
      throw std::invalid_argument("deallocate: Invalid pointer");
    } else if (bytes != i->_bytes) {
      throw std::invalid_argument("deallocate: Size mismatch");
    } else if (alignment != i->_alignment) {
      throw std::invalid_argument("deallocate: Alignment mismatch");
    }
    _upstream->deallocate(p, i->_bytes, i->_alignment);
    _blocks.erase(i);
    _bytes_outstanding -= bytes;
  }
  bool do_is_equal(const pmr::memory_resource &other) const noexcept override {
    // convert for printing
    auto this_ptr{reinterpret_cast<uint64_t>(this)};
    auto other_ptr{reinterpret_cast<uint64_t>(&other)};
    std::cout << std::format("(do_is_equal :this_ptr '{}' :other_ptr '{}')\n",
                             this_ptr, other_ptr);
    return this == &other;
  }

private:
  struct allocation_rec {
    void *_ptr;
    size_t _bytes;
    size_t _alignment;
  };
  ;
  pmr::memory_resource *_upstream{};
  size_t _bytes_allocated{};
  size_t _bytes_outstanding{};
  size_t _bytes_highwater{};
  pmr::vector<allocation_rec> _blocks{};
  static size_t _s_leaked_bytes;
  static size_t _s_leaked_blocks;
};
struct point_2d {
  double x;
  double y;
};
;

int main(int argc, char **argv) {
  constexpr int rawN{1'600};
  auto raw{std::array<std::byte, rawN>()};
  auto buf0{pmr::monotonic_buffer_resource(raw.data(), raw.size(),
                                           pmr::null_memory_resource())};
  auto buf{test_resource(&buf0)};
  constexpr int nPoints{100};
  auto sizeof_point_2d{sizeof(point_2d)};
  std::cout << std::format(
      "(main :rawN '{}' :nPoints '{}' :sizeof_point_2d '{}')\n", rawN, nPoints,
      sizeof_point_2d);
  auto points{pmr::vector<point_2d>(nPoints, &buf)};
  points[0] = {0.10F, 0.20F};
  return 0;
}
