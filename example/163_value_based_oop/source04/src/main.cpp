#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
// experiments with type erasure and strategy design patterns
class UniversalTE {
  class Interface {
  public:
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
  };
  std::unique_ptr<Interface> pimpl;
  template <typename Type> struct is_shared_ptr : std::false_type {};
  template <typename Type>
  struct is_shared_ptr<std::shared_ptr<Type>> : std::true_type {};
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
    template <typename Object3529, typename Strategy3530>
    Implementation(Object3529 &&object3529, Strategy3530 &&strategy3530)
        : object_{std::forward<Object3529>(object3529)},
          strategy_{std::forward<Strategy3530>(strategy3530)} {}
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
            *other.pimpl)} {
    std::cout << "(copy constructor" << ")\n";
  }
  // copy assignment operator
  template <typename Object, typename Strategy>
  UniversalTE &operator=(const UniversalTE &other) {
    std::cout << "(copy assignment operator" << ")\n";
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
            std::move(*other.pimpl))} {
    std::cout << "(move constructor" << ")\n";
  }
  // move assignment operator
  template <typename Object, typename Strategy>
  UniversalTE &operator=(UniversalTE &&other) {
    std::cout << "(move assignment operator" << ")\n";
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
  void meow() const {
    std::cout << "(meow" << std::format(":name '{}')\n", name);
  }
  void scratch() const {
    std::cout << "(scratch" << std::format(":name '{}')\n", name);
  }
};
class Dog {
public:
  std::string name;
  Dog(std::string_view name) : name{name} {}
  void bark() const {
    std::cout << "(bark" << std::format(":name '{}')\n", name);
  }
  void sit() const {
    std::cout << "(sit" << std::format(":name '{}')\n", name);
  }
  void waggle() const {
    std::cout << "(waggle" << std::format(":name '{}')\n", name);
  }
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
  auto kurt{std::make_shared<Cat>("kurt")};
  auto rover{Dog("rover")};
  auto s1{PetStrategy1()};
  auto ss1{std::make_shared<PetStrategy1>()};
  UniversalTE k1{kurt, s1};
  UniversalTE copy_k1{k1};
  UniversalTE move_k1{std::move(k1)};
  UniversalTE l1{lazy, ss1};
  UniversalTE r1{rover, s1};
  std::vector<UniversalTE> v;
  v.emplace_back(lazy, s1);
  v.emplace_back(rover, s1);
  v.emplace_back(kurt, ss1);
  for (auto &&e : v) {
    e.getTreat();
    e.getPetted();
  }
  return 0;
}
