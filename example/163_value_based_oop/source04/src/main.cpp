#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
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
    template <typename Object510, typename Strategy511>
    Implementation(Object510 &&object510, Strategy511 &&strategy511)
        : object_{std::forward<Object510>(object510)},
          strategy_{std::forward<Strategy511>(strategy511)} {}
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
  UniversalTE(UniversalTE const &other)
      : pimpl{std::make_unique<Implementation<std::__remove_cvref_t<Object>,
                                              std::__remove_cvref_t<Strategy>>>(
            *other.pimpl)} {}
  // copy assignment operator
  template <typename Object, typename Strategy>
  UniversalTE &operator=(UniversalTE const &other) {
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
  void meow() { std::cout << std::format("(meow)\n"); }
};
class PetStrategy1 {
public:
  void getTreat(const Cat &cat) { cat.meow(); }
};

int main() {
  auto lazy{Cat("lazy")};
  auto s1{PetStrategy1()};
  auto v{std::vector < UniversalTE({lazy, s1})};
}
