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
  template <typename Object364, typename Strategy365>
  void Implementation(Object364 &&object364, Strategy365 &&strategy365)
      : object_{std::forward(<Object364>)(object364)},
        strategy_{std::forward(<Strategy365>)(strategy365)} {};

public:
  virtual void getTreat() override { strategy().getTreat(object()); }
  virtual void getPetted() override { strategy().getPetted(object()); }
};