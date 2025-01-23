#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
class UniversalTE {
  class Interface {
  public:
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
  };
  std::unique_ptr<Interface> pimpl;
  template <typename Type> struct is_shared_ptr : std::false_type {};
  template <typename Type> struct is_shared_ptr<std::shared_ptr<Type>> {};
  template <typename Object, typename Strategy>
  class Implementation final : public Interface {
    Object object_;
    Strategy strategy_;

  public:
    auto &object() {
      if constexpr (is_shared_ptr<std::__remove_cvref_t<Object>>::value) {
        return *object_;
      } else {
        return object_;
      };
    }
    auto &strategy() {
      if constexpr (is_shared_ptr<std::__remove_cvref_t<Strategy>>::value) {
        return *strategy_;
      } else {
        return strategy_;
      };
    }
    template <typename Object1478, typename Strategy1479>
    Implementation(Object1478 &&object1478, Strategy1479 &&strategy1479)
        : object_{std::forward<Object1478>(object1478)},
          strategy_{std::forward<Strategy1479>(strategy1479)} {}
    void getTreat() override { strategy().getTreat(object()); }
    void getPetted() override { strategy().getPetted(object()); };
  };

public:
  template <typename Object, typename Strategy>
  UniversalTE(Object object_, Strategy strategy_)
      : pimpl{std::make_unique<Implementation<std::__remove_cvref_t<Object>,
                                              std::__remove_cvref_t<Strategy>>>(
            std::forward<Object>(object_),
            std::forward<Strategy>(strategy_))} {};
  // copy and move constructors
  // copy constructor
  template <typename Object, typename Strategy>
  UniversalTE(const UniversalTE &other)
      : pimpl{std::make_unique<Implementation<std::__remove_cvref_t<Object>,
                                              std::__remove_cvref_t<Strategy>>>(
            *other.pimpl)} {}
  // copy assignment operator
  template <typename Object, typename Strategy>
  UniversalTE &operator=(const UniversalTE &other) {
    if (this == &other) {
      return *this;
    }
    *pimpl = *other.pimpl;
    return *this;
  }
  // move constructor
  template <typename Object, typename Strategy>
  UniversalTE(UniversalTE &&other)
      : pimpl{std::make_unique<Implementation<std::__remove_cvref_t<Object>,
                                              std::__remove_cvref_t<Strategy>>>(
            std::move(*other.pimpl))} {}
  // move assignment operator
  template <typename Object, typename Strategy>
  UniversalTE &operator=(UniversalTE &&other) {
    if (this == &other) {
      return *this;
    }
    *pimpl = std::move(*other.pimpl);
    return *this;
  }
  void getTreat() const { pimpl->getTreat(); }
  void getPetted() const { pimpl->getPetted(); }
};
class Cat {
public:
  std::string name;
  Cat(std::string_view name) : name{name} {}
  void meow() const { std::cout << "(meow" << ")\n"; }
  void scratch() const { std::cout << "(scratch" << ")\n"; }
};
class Dog {
public:
  std::string name;
  Dog(std::string_view name) : name{name} {}
  void bark() const { std::cout << "(bark" << ")\n"; }
  void sit() const { std::cout << "(sit" << ")\n"; }
  void waggle() const { std::cout << "(waggle" << ")\n"; }
};
class PetStrategy1 {
public:
  void getTreat(const Cat &cat) {
    cat.meow();
    cat.scratch();
  }
  void getPetted(const Cat &cat) { cat.meow(); }
  void getPetted(const Dog &dog) { dog.waggle(); }
  void getTreat(const Dog &dog) { dog.sit(); }
};

int main() {
  auto lazy{Cat("lazy")};
  auto rover{Dog("rover")};
  auto s1{PetStrategy1()};
  UniversalTE l1{lazy, s1};
  UniversalTE r1{rover, s1};
  std::vector<UniversalTE> v;
  v.emplace_back(lazy, s1);
  v.emplace_back(rover, s1);
  for (auto &&e : v) {
    e.getTreat();
    e.getPetted();
  }
  return 0;
}
