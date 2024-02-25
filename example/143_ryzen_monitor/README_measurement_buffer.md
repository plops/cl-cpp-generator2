# Intro

- currently i use a deque to store the data in a FIFO for plotting. i
  don't think this is optimal
  
- this is an attempt to get something more performant

- question to gemini advanced:

Q: i need a C++ implementation, ideally using a datastructure from STL
or a library. it is important that this is as performant as
possible. deque is not cacheable. if you implement your own container,
please keep the interface similar to deque, though. i need the methods
.push_back, .pop_front and .at and possibly operator[]


## answer by gemini

You're absolutely right! Here's a C++ implementation tailored for your measurement points scenario with a focus on performance and cache-friendliness.

**Implementation: Using a Circular Buffer**

```c++
#include <vector>
#include <algorithm>

template <typename T>
class MeasurementBuffer {
public:
    MeasurementBuffer(size_t max_size) : data_(max_size), head_(0), size_(0) {}

    void push_back(const T& value) {
        data_[head_] = value;
        head_ = (head_ + 1) % data_.size();
        size_ = std::min(size_ + 1, data_.size());  
    }

    T pop_front() {
        if (empty()) throw std::runtime_error("Buffer is empty"); 
        T ret = data_[head_];
        head_ = (head_ - 1 + data_.size()) % data_.size(); 
        size_--;
        return ret;
    }

    T& at(size_t index) {
        if (index >= size_) throw std::out_of_range("Index out of bounds");
        return data_[(head_ + index) % data_.size()];
    }

    T& operator[](size_t index) {
        return at(index);
    }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

private:
    std::vector<T> data_;
    size_t head_; 
    size_t size_; 
}; 
```

**Explanation:**

1. **Circular Buffer:** The core concept is a circular buffer using an `std::vector` to hold the data. This ensures contiguous memory allocation, maximizing cache efficiency.
2. **`head_` and `size_`:**  The `head_` index points to where the next element will be inserted. `size_` tracks the current number of elements.
3. **Wrapping Behavior:** The modulo operations (%) in `push_back` and `pop_front` create the wrap-around behavior.
4. **Interface:** Provides `push_back`, `pop_front`, `at`, `operator[]`, `empty`, and `size` methods,  mimicking a deque-like interface.

**How to Use:**

```c++
#include <iostream>

int main() {
    MeasurementBuffer<double> buffer(200); // Store up to 200 temperature values

    // Simulate acquiring measurements
    for (int i = 0; i < 250; ++i) {
        buffer.push_back(i * 0.5); // Simulate temperature
    }

    // Access and print
    for (size_t i = 0; i < buffer.size(); ++i) {
        std::cout << buffer[i] << " ";
    } 
    std::cout << std::endl;
} 
```

**Let me know if you'd like any modifications to the interface or have further optimization requirements in mind!**
