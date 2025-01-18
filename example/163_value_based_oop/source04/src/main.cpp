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
  <typename Object, typename Strategy> class Implementation
      : public Interface() {
  private:
    Object object_;
    Strategy strategy_;

  public:
    void object() {
      if (constexpr is_shared_ptr<std::remove_cvref_t<Object>>::value) {
        return *object_;
      } else {
        return object;
      }
    }
    void strategy() {
      if (constexpr is_shared_ptr<std::remove_cvref_t<Strategy>>::value) {
        return *strategy_;
      } else {
        return strategy;
      }
    }
  };

public:
};