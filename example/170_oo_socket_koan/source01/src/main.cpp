#include <ISocket.h>
#include <TCPSocket.h>
#include <UDPSocket.h>
#include <iostream>
#include <memory>

#include <array>
#include <vector>
#include <functional>
using namespace std;

template <typename T = uint8_t> class IPointer {
public:
  virtual ~IPointer() = default;
  virtual T deref() = 0;
  virtual bool isNull() = 0;
};

template <typename T = uint8_t> //, typename... Params>
class NonNullPointer : public IPointer<T> {
  public:
  // explicit NonNullPointer(Params&&... args)
  // :ptr{make_unique<T>(forward<Params>(args)...)} {
  // }
  explicit NonNullPointer(T* out)
    :ptr{out} {
  }
  ~NonNullPointer() = default;
  T deref() override {
    return *ptr;
  }
  bool isNull() override { return false;}
  private:
  // unique_ptr<T> ptr;
  T* ptr;
};

int fun() {
  array<int, 3> a{1, 2, 3};
  // array<int *, 3> b{&a[0], &a[1], &a[2]};
  // int &r = a[0];
  // int &ra[3] = {r, r, r};
  NonNullPointer<int> r{&a[0]};
  // array<NonNullPointer<int,int>,3> ra{{&a[0],&a[1],&a[2]}};
  return r.deref();
}

template <typename T>
class IPool {
  public:
  virtual ~IPool() noexcept(false) = default;
  virtual T* next() = 0;
};

template <typename C>
class MyIterator { // Stroustrup 2022 p 174
public:
  using value_type = typename C::value_type;
  using difference_type = int;

  MyIterator() = default;
  explicit MyIterator(C&cc) : pc(&cc){}
  // MyIterator(C&cc,typename C::iterator pp): pc(&cc), p(pp){}
  MyIterator& operator++() {++pc; return *this;}
  MyIterator operator++(int) {auto tmp=*this; ++pc; return tmp;}
  value_type& operator*() const {return *pc;}

  bool operator==(const MyIterator& rhs) const {return pc == rhs.pc;}
  bool operator!=(const MyIterator& rhs) const {return pc != rhs.pc;}
private:
  C*pc{}; // Pointer to container, default nullptr
  // typename C::iterator p{pc->data()};
};


template <typename T>
class IArray {
public:
  virtual ~IArray() noexcept(false) = default;
  virtual T aref(size_t index) = 0;
  virtual T* data() = 0;
  // virtual MyIterator<IArray> begin() = 0;
  // virtual MyIterator<IArray> end() = 0;
  virtual size_t size() = 0;
};


template <typename T>
class NormalArray final : public IArray<T> {
public:
  using value_type = T;
  explicit NormalArray(size_t capacity = 1024)
  : capacity_{capacity}, array{new T[capacity_]} {
    cout << "Normal array created" << endl;
    fill(array, array + capacity_, value_type(0));
  }
  ~NormalArray() override {
    cout << "Normal array destroyed" << endl;
    delete[] array;
  }
  T aref(size_t index) override {
    cout << "NormalArray::aref " << index << endl;
    if (index<capacity_)
      return array[index];
    return static_cast<T>(0);
  }
  T* data() override {return array;}
  // FIXME: how to povide interface so that range-based for loop works?
  //  MyIterator<NormalArray> begin()  {return MyIterator<NormalArray>(array);}
  // MyIterator<NormalArray> end()  {return MyIterator<NormalArray>(array + size());}
  size_t size() override {return capacity_;}
private:
  size_t capacity_;
  T* array;
};

template <typename T>
class Pool final : public IPool<T> {
  public:
  explicit Pool(size_t capacity, function<unique_ptr<T>()> element_creator =[](){cout << "Default Pool Filler"  << endl; return make_unique<T>();}) {
    cout << "Pool created" << endl;
    vec.reserve(capacity);
    for (size_t i = 0; i < capacity; ++i) {
      vec.emplace_back(element_creator());
    }
    current = vec.begin();
  }
  ~Pool() noexcept(false) override {
    cout << "Pool destroyed" << endl;
  };
  T* next() override {
    // if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
    //   cout << "Pool next idx=" << current-vec.begin() << " val=" << *current <<  endl;
    // } else {
    //   cout << "Pool next idx=" << current-vec.begin() << " array=[" << (current->get()->aref(0)) << "," << (current->get()->aref(1)) << "...]" << endl;
    // }
    auto* val = current->get();
    ++current;
    if (current == vec.end()) {
      current = vec.begin();
    }
    return val;
  }
private:
  vector<unique_ptr<T>> vec;
  decltype(vec.begin()) current;
};

void fun2() {
  NormalArray<int> a(7);
  a.data()[1]=1;
  cout << a.aref(0) << endl;
  cout << a.aref(1) << endl;

  // for (const auto&e : a)
  //   cout << e << endl;

  cout << "Pool<float>" << endl;
  auto p1 = Pool<float>(3);
  auto q= p1.next();
  cout << q << endl;
  *q=3.13f;
  cout << *q << endl;
  auto qq = p1.next();
  cout << *qq << endl;
  *qq = 3.2f;

  cout << "Pool<NormalArray<int>>" << endl;
  auto p2 = Pool<NormalArray<int>>(3,[](){return make_unique<NormalArray<int>>(7);});
  // auto qqq = p2.next();
  // qqq.data()[1] = 3;
  // cout << "qqq " << qqq.aref(1) << endl;

  cout << "Ptr IPool<NormalArray<float>>" << endl;

  IPool<IArray<float>>* pool = new Pool<NormalArray<float>>(2,[](){return make_unique<NormalArray<float>>(3);});
  delete pool;
}
int main(int argc, char *argv[]) {
  // cout << fun() << endl;
  fun2();
  return 0;
}
/*
{
  if (argc < 2) {
    perror("Too few arguments");
    return EXIT_FAILURE;
  }
  const string protocolChoice = argv[1];

  unique_ptr<ISocket> currentSocket;

  if (protocolChoice == "tcp") {
    currentSocket = make_unique<TCPSocket>();
  } else if (protocolChoice == "udp") {
    currentSocket = make_unique<UDPSocket>();
  } else {
    perror("Unknown protocol choice");
    return EXIT_FAILURE;
  }

  cout << "Enter port: " << endl;
  uint16_t port;
  cin >> port;

  if (!currentSocket->open(port)) {
    perror("Failed to open socket");
    return EXIT_FAILURE;
  }

  string command;
  cout << "Enter command (send/receive/close): " << endl;
  cin >> command;

  if (command == "send") {
    string message;
    cout << "Enter message: " << endl;
    cin >> message;
    cout << "Sending: " << message << "..." << endl;
    if (!currentSocket->send(message)) {
      perror("Send failed");
    } else if (command == "receive") {
      if (const string receivedData = currentSocket->receive();
          !receivedData.empty()) {
        cout << "Received: " << receivedData << endl;
      }
    } else if (command != "close") {
      perror("Unknown command");
    }
  }
  currentSocket->close();
  return EXIT_SUCCESS;
}
*/