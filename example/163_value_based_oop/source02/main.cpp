#include <memory>

struct Interface {
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
};


class StatelessTE {
private:
    std::unique_ptr<Interface> pimpl;
    public:
        template<typename Object, typename Strategy>
        StatelessTE(Object&& object, Strategy&& strategy)
            : pimpl(std::make_unique<Implementation<std::remove_cvref_t<Object>,
                std::remove_cvref_t<Strategy>>>(
                std::forward<Object>(object),
                                                  std::forward<Strategy>(strategy)))
        {}
    void getTreat() {pimpl->getTreat();}
    void getPetted() {pimpl->getPetted();}
        };

int main()
{
    StatelessTE em{};
    return 0;
}