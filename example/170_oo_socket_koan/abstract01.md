I want to create a generic pool object IPool<IArray<...>> that is generic in two aspects: the pool implementation and the array implementation. i'm confused how to do that. help me, start with the code that i currently have:

```c++
#include <array>
#include <vector>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>

using namespace std;

// Forward declarations
template <typename T> class IArray;
template <typename T> class IPool;


// ============ IPointer ============
template <typename T = uint8_t> class IPointer {
public:
  virtual ~IPointer() = default;
  virtual T deref() = 0;
  virtual bool isNull() = 0;
};

template <typename T = uint8_t>
class NonNullPointer : public IPointer<T> {
  public:
  explicit NonNullPointer(T* out)
    :ptr{out} {
      if (!out) {
          throw std::runtime_error("NonNullPointer cannot be initialized with a null pointer");
      }
  }
  ~NonNullPointer() = default;
  T deref() override {
    return *ptr;
  }
  bool isNull() override { return false;}
  private:
  T* ptr;
};


// ============ IArray ============

template <typename T>
class IArray {
public:
  virtual ~IArray() noexcept = default;
  virtual T& operator[](size_t index) = 0;  // More idiomatic array access
  virtual const T& operator[](size_t index) const = 0; // const version
  virtual T* data() = 0;
  virtual const T* data() const = 0;
  virtual size_t size() const = 0;
};

template <typename T>
class NormalArray final : public IArray<T> {
public:
  using value_type = T;
  explicit NormalArray(size_t capacity = 1024)
    : capacity_{capacity}, array{std::make_unique<T[]>(capacity_)}  // Use unique_ptr
  {
    std::cout << "Normal array created" << std::endl;
    std::fill(array.get(), array.get() + capacity_, value_type(0)); // Use .get()
  }
  ~NormalArray() override {
    std::cout << "Normal array destroyed" << std::endl;
  }

  T& operator[](size_t index) override {
    if (index >= capacity_) {
        throw std::out_of_range("Index out of bounds in NormalArray");
    }
      return array[index];
  }

    const T& operator[](size_t index) const override {
    if (index >= capacity_) {
        throw std::out_of_range("Index out of bounds in NormalArray");
    }
      return array[index];
  }

  T* data() override { return array.get(); }  // Return raw pointer from unique_ptr
  const T* data() const override { return array.get(); }
  size_t size() const override { return capacity_; }

private:
  size_t capacity_;
  std::unique_ptr<T[]> array;  // Use unique_ptr for automatic memory management
};


// ============ IPool ============

template <typename T>
class IPool {
public:
  virtual ~IPool() noexcept = default;
  virtual T* next() = 0;  // Return a pointer to the next available object
  virtual size_t capacity() const = 0; // Added capacity
};


template <typename ArrayType>
class Pool final : public IPool<ArrayType> {
    static_assert(std::is_base_of<IArray<typename ArrayType::value_type>, ArrayType>::value,
                "ArrayType must derive from IArray");
public:
    using value_type = typename ArrayType::value_type; // get the type that the array holds

  explicit Pool(size_t capacity, std::function<std::unique_ptr<ArrayType>()> array_creator)
      : capacity_(capacity), current_index_(0) {
    std::cout << "Pool created" << std::endl;
    arrays_.reserve(capacity);
    for (size_t i = 0; i < capacity; ++i) {
      arrays_.emplace_back(array_creator());
    }
  }
  ~Pool() noexcept override {
    std::cout << "Pool destroyed" << std::endl;
  }

    ArrayType* next() override {
      ArrayType* arr = arrays_[current_index_].get();
        current_index_ = (current_index_ + 1) % capacity_; // Wrap around
        return arr;
    }
    size_t capacity() const override {return capacity_;}

private:
  size_t capacity_;
    size_t current_index_;
  std::vector<std::unique_ptr<ArrayType>> arrays_;
};


// ============ Generic IPool (the core of your question) ============

template <template <typename> class PoolImpl, template<typename> class ArrayImpl, typename T>
class GenericPool {
public:

    using ArrayType = ArrayImpl<T>;
    using PoolType = PoolImpl<ArrayType>;

    // Constructor
    GenericPool(size_t pool_capacity, size_t array_capacity) :
      pool_(pool_capacity, [array_capacity]() { return std::make_unique<ArrayType>(array_capacity); })
    {}

      // Delegate next() to the underlying pool
      ArrayType* next() {
        return pool_.next();
      }
      size_t pool_capacity() const {
        return pool_.capacity();
      }

private:
    PoolType pool_;
};


void test_generic_pool() {
    std::cout << "\n--- test_generic_pool ---\n";

    // Create a GenericPool with Pool as the pool implementation and NormalArray as the array implementation
    GenericPool<Pool, NormalArray, float> generic_pool(3, 5);

    // Get a few arrays from the pool
    auto* arr1 = generic_pool.next();
    auto* arr2 = generic_pool.next();
    auto* arr3 = generic_pool.next();

    // Modify the arrays
    (*arr1)[0] = 1.1f;
    (*arr1)[1] = 2.2f;
    (*arr2)[0] = 3.3f;
    (*arr3)[4] = 4.4f;

    // Get arr1 again (should be the same as the first arr1 we got)
    auto* arr1_again = generic_pool.next();
    std::cout << "arr1[0]: " << (*arr1_again)[0] << std::endl; // Output should be 1.1
    std::cout << "arr1[1]: " << (*arr1_again)[1] << std::endl; // Output should be 2.2
    std::cout << "arr2[0]: " << (*arr2)[0] << std::endl;       // Output should be 3.3
    std::cout << "arr3[4]: " << (*arr3)[4] << std::endl;       // Output should be 4.4
     std::cout << "pool capacity: " << generic_pool.pool_capacity() << std::endl;
    //Demonstrating const correctness:
    const auto* arr1_const = arr1;
    std::cout << "arr1_const[0]: " << (*arr1_const)[0] << std::endl;

    // Test with integers
    GenericPool<Pool, NormalArray, int> int_pool(2, 4);
    auto* int_arr1 = int_pool.next();
    (*int_arr1)[2] = 123;
    std::cout << "int_arr1[2]: " << (*int_arr1)[2] << std::endl;
}


void test_basic_pool_and_array() {
    std::cout << "\n--- test_basic_pool_and_array ---\n";

    // Test NormalArray
    NormalArray<int> intArray(3);
    intArray[0] = 10;
    intArray[1] = 20;
    intArray[2] = 30;
    std::cout << "intArray[1]: " << intArray[1] << std::endl;

    // Test Pool with NormalArray
    Pool<NormalArray<float>> floatArrayPool(2, []() { return std::make_unique<NormalArray<float>>(4); });
    auto* floatArray1 = floatArrayPool.next();
    (*floatArray1)[0] = 1.0f;
    (*floatArray1)[3] = 4.0f;
    auto* floatArray2 = floatArrayPool.next();
    (*floatArray2)[0] = 5.0f;
    auto* floatArray1Again = floatArrayPool.next();  // Get floatArray1 again
    std::cout << "floatArray1Again[0]: " << (*floatArray1Again)[0] << std::endl;
    std::cout << "floatArray1Again[3]: " << (*floatArray1Again)[3] << std::endl;
    std::cout << "floatArray2[0]: " << (*floatArray2)[0] << std::endl;
}

int main() {
    test_basic_pool_and_array();
    test_generic_pool();

     // Demonstrating exception handling
    try {
        NormalArray<int> intArray(3);
        intArray[5] = 10;  // Out of bounds
    } catch (const std::out_of_range& e) {
        std::cerr << "Caught expected out_of_range exception: " << e.what() << std::endl;
    }
    try {
        NormalArray<int> intArray(3);
        NonNullPointer<int> nullPtr(nullptr);
        
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught expected runtime_error exception: " << e.what() << std::endl;
    }
    return 0;
}
```

Key changes and explanations:

1.  **`GenericPool` Class:**
    *   This is the core of the solution.  It takes *template template parameters* `PoolImpl` and `ArrayImpl`.  This allows you to specify the pool and array implementations as template arguments.
    *   `using ArrayType = ArrayImpl<T>;` and `using PoolType = PoolImpl<ArrayType>;` define convenient aliases for the specific array and pool types being used.
    *   The constructor takes the pool capacity and array capacity, and uses a lambda to create the array instances within the pool's initialization.  This is much cleaner than passing a separate array creation function.
    *   `next()` simply delegates to the underlying `pool_` member's `next()` method.  This ensures that the `GenericPool` uses the correct pool implementation.

2.  **`IPool` and `Pool` Changes:**
    *   `IPool<T>` now takes the array type `T` as a template parameter (e.g., `IPool<NormalArray<float>>`).
    *   `Pool` is now `Pool<ArrayType>`, where `ArrayType` must be derived from `IArray`.  This enforces the constraint that the pool manages arrays that implement the `IArray` interface.
        *   A `static_assert` is used to check at *compile time* that `ArrayType` is indeed derived from `IArray`. This provides early error detection.
        * using `value_type` to get the type that array stores.
    *  `next()` in `Pool` returns a pointer to the `ArrayType` (e.g., `NormalArray<float>*`).  The `current_index_` logic handles wrapping around the pool.
    *  Added `capacity()` member function to `IPool` and `Pool`.

3.  **`IArray` and `NormalArray` Changes:**
    *   `IArray` now has `operator[]` for both read and write access (const and non-const versions).  This is the standard way to provide array-like access in C++.
    *   `NormalArray` uses `std::unique_ptr<T[]>` for managing the array's memory.  This *automatically* handles deallocation, preventing memory leaks.  This is crucial for RAII (Resource Acquisition Is Initialization).
    *   `data()` in `NormalArray` returns `array.get()`, which gives you the raw pointer managed by the `unique_ptr`.
    *   Added const versions of `data()` and `operator[]`.
    * added error checking for index out of bounds in `operator[]`.

4.  **Test Functions:**
    *   `test_basic_pool_and_array()`: Tests the basic functionality of `NormalArray` and `Pool` independently.
    *   `test_generic_pool()`:  Demonstrates the use of `GenericPool`.  It creates a `GenericPool` using `Pool` and `NormalArray`, gets arrays, modifies them, and verifies that the pool is working correctly.  This is the key test case to show that your generic solution works.

5.  **RAII (Resource Acquisition Is Initialization):**
    *   The use of `std::unique_ptr` throughout the code (in `NormalArray` and `Pool`) is a key example of RAII.  Resources (memory) are automatically released when the owning objects go out of scope.  This is a fundamental principle of modern C++ for preventing memory leaks and making code more robust.

6.  **Exception Handling:**
    *   Added a `try-catch` block in `main` to demonstrate exception handling for out-of-bounds access in `NormalArray` and for null pointer.

7. **`NonNullPointer` Improvements**:
    * added check and exception if someone tries to initialize the `NonNullPointer` with null pointer.

8. **Const correctness:**
    *  Added const versions of `data()` and `operator[]` in `IArray` and `NormalArray` and also added example of using const version.

How to compile and run (using g++):

```bash
g++ -std=c++17 your_file.cpp -o your_program
./your_program
```

This improved code is significantly more robust, easier to use, and demonstrates the correct way to create a generic pool in C++. It leverages modern C++ features like `unique_ptr`, template template parameters, and `static_assert` to create a clean, type-safe, and memory-safe solution.  The tests also thoroughly demonstrate the intended functionality.  The code is also well-commented to explain the purpose of each part.
3