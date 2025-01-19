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
  template <typename Object, typename Strategy>
  class Implementation : public Interface() {
  private:
    Object object_;
    Strategy strategy_;

  public:
    void object() {
      if constexpr (is_shared_ptr<std::remove_cvref_t<Object>>::value) {
        return *object_;
      } else {
        return object_;
      };
    }
    void strategy() {
      if constexpr (is_shared_ptr<std::remove_cvref_t<Strategy>>::value) {
        return *strategy_;
      } else {
        return strategy_;
      };
    }
  };
  template <typename Object334, typename Strategy335>
  void Implementation(Object334 &&object334, Strategy335 &&strategy335)
      : object_{std::forward(<Object334>)(object334)},
        strategy_{std::forward(<Strategy335>)(strategy335)} {};

public:
  void getTreat() override { strategy().getTreat(object()); }
  void getPetted() override { strategy().getPetted(object()); }
};