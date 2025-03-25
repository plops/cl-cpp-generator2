#include <ISocket.h>
#include <TCPSocket.h>
#include <UDPSocket.h>
#include <iostream>
#include <memory>

#include <array>
#include <vector>
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
  virtual T& next() = 0;
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
  explicit NormalArray(size_t capacity)
  : capacity_{capacity}, array{new T[capacity_]} {
    fill(array, array + capacity_, value_type(0));
  }
  ~NormalArray() override {
    delete[] array;
  }
  T aref(size_t index) override {
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
  explicit Pool(size_t capacity) {
    cout << "Pool created" << endl;
    vec.resize(capacity);
    current = vec.begin();
  }
  ~Pool() noexcept(false) override {
    cout << "Pool destroyed" << endl;
  };
  T& next() override {
    cout << "Pool next idx=" << current-vec.begin() << " val=" << *current <<  endl;
    auto& val = *current;
    ++current;
    if (current == vec.end()) {
      current = vec.begin();
    }
    return val;
  }
private:
  vector<T> vec;
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
  q=3.13;
  cout << q << endl;
  auto qq = p1.next();
  cout << qq << endl;
  qq = 3.2;


  // IPool<NormalArray<uint8_t>>* pool = new Pool<NormalArray<uint8_t>>(8);
}
int main(int argc, char *argv[]) {
  // cout << fun() << endl;
  fun2();
  return 0;
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
