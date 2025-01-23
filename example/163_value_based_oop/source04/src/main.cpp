#include <memory>
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
    template <typename Object360, typename Strategy361>
    Implementation(Object360 &&object360, Strategy361 &&strategy361)
        : object_{std::forward<Object360>(object360)},
          strategy_{std::forward<Strategy361>(strategy361)} {}
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
  UniversalTE(UniversalTE const &other)
      : pimpl{std::make_unique<Interface>(*other.pimpl)} {}
  // copy assignment operator
  UniversalTE &operator=(UniversalTE const &other) {
    if (this == &other) {
      return *this;
    }
    *pimpl = *other.pimpl;
    return *this;
  }
  // move constructor
  UniversalTE(UniversalTE &&other)
      : pimpl{std::make_unique<Interface>(std::move(*other.pimpl))} {}
  // move assignment operator
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

int main() {}
