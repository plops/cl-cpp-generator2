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
  std::unique_ptr<Interface> const pimpl;
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
    template <typename Object828, typename Strategy829>
    Implementation(Object828 &&object828, Strategy829 &&strategy829)
        : object_{std::forward<Object828>(object828)},
          strategy_{std::forward<Strategy829>(strategy829)} {}
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
class PetStrategy1 {
public:
  void getTreat(const Cat &cat) {
    cat.meow();
    cat.scratch();
  }
  void getPetted(const Cat &cat) { cat.meow(); }
};

int main() {
  auto lazy{Cat("lazy")};
  auto s1{PetStrategy1()};
  UniversalTE e{lazy, s1};
  std::vector<UniversalTE> v;
  v.emplace_back(e);
}
