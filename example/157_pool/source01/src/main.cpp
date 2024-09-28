#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <string>
using namespace std;
class test_resource : public pmr::memory_resource {
public:
  test_resource(pmr::memory_resource *parent) {}
  ~test_resource() {}
  pmr::memory_resource *upstream() const {}
  size_t bytes_allocated() const {}
  size_t bytes_deallocated() const {}
  size_t bytes_outstanding() const {}
  size_t bytes_highwater() const {}
  size_t blocks_outstanding() const {}
  // We can't throw in the destructor that is why we need the following three
  // functions
  size_t leaked_bytes() {}
  size_t leaked_blocks() {}
  void clear_leaked() {}

protected:
  void *do_allocate(size_t bytes, size_t alignment) override {}
  void do_deallocate(void *p, size_t bytes, size_t alignment) override {}
  bool do_is_equal(const pmr::memory_resource &other) const noexcept override {}
};

int main(int argc, char **argv) { return 0; }
