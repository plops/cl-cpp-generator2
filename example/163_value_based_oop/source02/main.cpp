#include <memory>
#include <vector>

struct Interface
{
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
};


template<typename Object, typename Strategy>
struct Implementation : public Interface
{
    Object object;
    Strategy strategy;

    template<typename Object2, typename Strategy2>
    Implementation(Object2&& o, Strategy2&& s)
        : object(std::forward<Object2>(o))
    , strategy(std::forward<Strategy2>(s))
    {}
    void getTreat() override
    {
        strategy.getTreat(object);
    }
    void getPetted() override
    {
        strategy.getPetted(object);
    }
};

struct StatelessTE
{
private:
    std::unique_ptr<Interface> pimpl;

public:
    template <typename Object, typename Strategy>
    StatelessTE(Object&& object, Strategy&& strategy) :
        pimpl(std::make_unique<Implementation<std::remove_cvref_t<Object>, std::remove_cvref_t<Strategy>>>(
            std::forward<Object>(object), std::forward<Strategy>(strategy)))
    {
    }
    void getTreat() { pimpl->getTreat(); }
    void getPetted() { pimpl->getPetted(); }
};


struct Cat
{
    void meow(){}
    void purr(){}
    void scratch(){}
};
struct Dog
{
    void bark(){};
    void sit(){};
    int hairsShedded{0};
    auto hairsSheddedCount(){ return hairsShedded; }
    void shed(){ hairsShedded += 1'000'000;}
};

struct PetStrategy1
{
    void getTreat(Cat& cat){ cat.meow(); cat.scratch(); }
    void getPetted(Cat& cat){ cat.purr(); }
    void getTreat(Dog& dog){ dog.sit(); }
    void getPetted(Dog& dog){ dog.bark(); dog.shed(); }
};

struct PetStrategy2
{
    int treatsSpent{   0};
    auto treatsSpentCount(){ return treatsSpent; }
    void getTreat(Cat& cat){ ++treatsSpent; cat.meow(); }
    void getPetted(Cat& cat){ cat.purr(); cat.purr(); }
    void getTreat(Dog& dog){ ++treatsSpent; dog.sit(); }
    void getPetted(Dog& dog){ dog.sit(); dog.shed(); }
};

int main()
{
    auto rover{Dog()};
    auto lazy{Cat()};
    auto s1{PetStrategy1()};
    auto s2{PetStrategy2()};
    // StatelessTE q0{rover, s1};
    // StatelessTE q1{rover, s2};
    // StatelessTE q2{lazy, s1};
    // StatelessTE q3{lazy, s2};
    // auto v{std::vector<StatelessTE>{q0,q1,q2,q3}};
    std::vector<StatelessTE> v;
    v.emplace_back(StatelessTE(rover, s2));
    v.emplace_back(StatelessTE(lazy, s1));

    for (auto&& e : v)
    {
        e.getTreat();
        e.getPetted();
    }
    return 0;
}
