#include <memory>
#include <vector>

struct Interface
{
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
};

template <typename Type>
struct is_shared_ptr : std::false_type
{
};
template <typename Type>
struct is_shared_ptr<std::shared_ptr<Type>> : std::true_type
{
};


// Object and Strategy get copied into the templated Implementation class (so we can't keep track of states in these objects)
template<typename Object, typename Strategy>
struct Implementation : public Interface
{
    Object object;
    Strategy strategy;

    // mimic Dog& operator=(...) {...; return *this; }

    auto& object_()
    {
        if constexpr (is_shared_ptr<std::remove_cvref_t<Object>>::value)
            return *object;
        else
            return object;
    }

    auto& strategy_()
        {
            if constexpr (is_shared_ptr<std::remove_cvref_t<Strategy>>::value)
                return *strategy;
            else
                return strategy;
            }


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



struct UniversalTE
{
private:
    std::unique_ptr<Interface> pimpl;

public:
    template <typename Object, typename Strategy>
    UniversalTE(Object&& object, Strategy&& strategy) :
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
    std::vector<UniversalTE> v;
    v.emplace_back(UniversalTE(rover, s2));
    v.emplace_back(UniversalTE(lazy, s1));

    for (auto&& e : v)
    {
        e.getTreat();
        e.getPetted();
    }
    return 0;
}
