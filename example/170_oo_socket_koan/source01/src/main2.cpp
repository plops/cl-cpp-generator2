#include <array>
#include <vector>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>

using namespace std;

// Forward declarations (important for circular dependencies)
template <typename T> class IArray;

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
  virtual T& operator[](size_t index) = 0;
  virtual const T& operator[](size_t index) const = 0;
  virtual T* data() = 0;
  virtual const T* data() const = 0;
  virtual size_t size() const = 0;
};

template <typename T>
class NormalArray final : public IArray<T> {
public:
  using value_type = T;
  explicit NormalArray(size_t capacity = 1024)
    : capacity_{capacity}, array{std::make_unique<T[]>(capacity_)}
  {
    std::cout << "Normal array created" << std::endl;
    std::fill(array.get(), array.get() + capacity_, value_type(0));
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

  T* data() override { return array.get(); }
  const T* data() const override { return array.get(); }
  size_t size() const override { return capacity_; }

private:
  size_t capacity_;
  std::unique_ptr<T[]> array;
};

// ============ IPool (Interface for Pools of IArrays) ============

template <typename T>
class IPool {
public:
    virtual ~IPool() = default;
    virtual IArray<T>* next() = 0; // Return a pointer to IArray<T>
    virtual size_t capacity() const = 0;
};

// ============ Concrete Pool Implementation ============

template <typename ArrayType>
class Pool final : public IPool<typename ArrayType::value_type> { // Inherit from IPool
    static_assert(std::is_base_of<IArray<typename ArrayType::value_type>, ArrayType>::value,
                  "ArrayType must derive from IArray");
public:
    using value_type = typename ArrayType::value_type;

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

    IArray<value_type>* next() override { // Return IArray<value_type>*
        IArray<value_type>* arr = arrays_[current_index_].get();
        current_index_ = (current_index_ + 1) % capacity_;
        return arr;
    }

    size_t capacity() const override { return capacity_; }

private:
    size_t capacity_;
    size_t current_index_;
    std::vector<std::unique_ptr<ArrayType>> arrays_;
};

// ============ GenericPool (Now a Concrete Class) ============

template <template <typename> class PoolImpl, template<typename> class ArrayImpl, typename T>
class GenericPool : public IPool<T> { // Inherit from IPool<T>
public:
    using ArrayType = ArrayImpl<T>;
    using PoolType = PoolImpl<ArrayType>;

    GenericPool(size_t pool_capacity, size_t array_capacity) :
      pool_(pool_capacity, [array_capacity]() { return std::make_unique<ArrayType>(array_capacity); })
    {}

    IArray<T>* next() override { // Return IArray<T>*
        return pool_.next();
    }

    size_t capacity() const override {
        return pool_.capacity();
    }

private:
    PoolType pool_; // Store the concrete pool
};


// ============ Example Usage and Polymorphism ============

class MyObject {
public:
    MyObject(IPool<float>* pool) : pool_(pool) {}

    void doSomething() {
        IArray<float>* arr = pool_->next();
        // ... use the array ...
        (*arr)[0] = 42.0f;
         std::cout << "MyObject using array, element 0: " << (*arr)[0] << std::endl;
    }

private:
    IPool<float>* pool_; // Store a pointer to the IPool interface
};

//another pool implementation
template <typename T>
class AnotherPool final : public IPool<T>{
public:
    AnotherPool(size_t capacity) : capacity_(capacity) {}

    IArray<T>* next() override {
       if (!arr) {
           arr =  std::make_unique<NormalArray<T>>(capacity_);
           return arr.get();
        }
        else {
           return arr.get();
        }
    }

    size_t capacity() const override { return capacity_; }

private:
    std::unique_ptr<NormalArray<T>> arr;
    size_t capacity_;
};

void test_polymorphic_pool() {
    std::cout << "\n--- test_polymorphic_pool ---\n";

    // Create a GenericPool using Pool and NormalArray
     GenericPool<Pool, NormalArray, float> generic_pool(3, 5);

    // Create another pool implementation
    AnotherPool<float> another_pool(7);

    // Create MyObject, passing a pointer to the IPool interface
    MyObject obj1(&generic_pool);
    MyObject obj2(&another_pool);


    // Call doSomething, which uses the pool polymorphically
    obj1.doSomething();
    obj1.doSomething();

     //changing pool in runtime:
    obj1 = MyObject(&another_pool);
    obj1.doSomething();
    obj2.doSomething();
}


void test_generic_pool() {
    std::cout << "\n--- test_generic_pool ---\n";
    GenericPool<Pool, NormalArray, float> generic_pool(3, 5);
    auto* arr1 = generic_pool.next();
    auto* arr2 = generic_pool.next();
      (*arr1)[0] = 1.1f;
    std::cout << "arr1[0]: " << (*arr1)[0] << std::endl;
    std::cout << "arr2[0]: " << (*arr2)[0] << std::endl;
}


void test_basic_pool_and_array() {
    std::cout << "\n--- test_basic_pool_and_array ---\n";
    NormalArray<int> intArray(3);
    intArray[0] = 10;
    std::cout << "intArray[0]: " << intArray[0] << std::endl;
    Pool<NormalArray<float>> floatArrayPool(2, []() { return std::make_unique<NormalArray<float>>(4); });
    auto* floatArray1 = floatArrayPool.next();
    (*floatArray1)[0] = 1.0f;
    auto* floatArray1Again = floatArrayPool.next();
    std::cout << "floatArray1Again[0]: " << (*floatArray1Again)[0] << std::endl;
}

int main() {
    test_basic_pool_and_array();
    test_generic_pool();
    test_polymorphic_pool();

    return 0;
}