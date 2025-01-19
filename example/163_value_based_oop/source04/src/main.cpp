#include <memory>
#include <type_traits>
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
    template <typename Object313, typename Strategy314>
    Implementation(Object313 &&object313, Strategy314 &&strategy314)
        : object_{std::forward<Object313>(object313)},
          strategy_{std::forward<Strategy314>(strategy314)} {}
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
  UniversalTE(const UniversalTE& other)
    : pimpl(other.pimpl)
  {
  }

  UniversalTE(UniversalTE&& other) noexcept
    : pimpl(std::move(other.pimpl))
  {
  }

  UniversalTE& operator=(const UniversalTE& other)
  {
    if (this == &other)
      return *this;
    pimpl = other.pimpl;
    return *this;
  }

  UniversalTE& operator=(UniversalTE&& other) noexcept
  {
    if (this == &other)
      return *this;
    pimpl = std::move(other.pimpl);
    return *this;
  }

  void getTreat() const { pimpl->getTreat(); }
  void getPetted() const { pimpl->getPetted(); }
};

int main() {}
