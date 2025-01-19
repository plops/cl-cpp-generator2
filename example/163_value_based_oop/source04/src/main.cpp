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
  class Implementationation final : public Interface {
  private:
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
    template <typename Object266, typename Strategy267>
    Implementation(Object266 &&object266, Strategy267 &&strategy267)
        : object_{std::forward<Object266>(object266)},
          strategy_{std::forward<Strategy267>(strategy267)} {}
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

int main() {}
