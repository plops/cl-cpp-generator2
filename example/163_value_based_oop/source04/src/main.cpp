#include <format>
#include <iostream>
#include <memory>
#include <vector>
class UniversalTE {
private:
  std::unique_ptr<Interface> pimpl;
  class Interface {
  public:
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
  };
  template <typename Type> struct is_shared_ptr : std::false_type {};
  template <typename Type> struct is_shared_ptr<std::shared_ptr<Type>> {};

public:
};