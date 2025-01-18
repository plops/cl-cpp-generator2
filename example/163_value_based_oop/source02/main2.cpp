#include <memory>
#include <vector>

struct Interface
{
    virtual ~Interface() = default;
    virtual void getTreat() = 0;
    virtual void getPetted() = 0;
};


// As opposode to main.cpp here Object and Strategy are stored as a point in the templated Implementation class
// Now the object or strategy can collect statistics (or any other state)

template <typename Object, typename Strategy>
struct Implementation : public Interface
{
    std::shared_ptr<Object> object;
    std::shared_ptr<Strategy> strategy;

    template <typename Object2, typename Strategy2>
    Implementation(Object2&& o, Strategy2&& s) : object(std::forward<Object2>(o)), strategy(std::forward<Strategy2>(s))
    {
    }
    void getTreat() override { strategy->getTreat(*object); }
    void getPetted() override { strategy->getPetted(*object); }
};

template <typename Type>
struct is_shared_ptr : std::false_type
{
};
template <typename Type>
struct is_shared_ptr<std::shared_ptr<Type>> : std::true_type
{
};

struct StatefulTE
{
private:
    std::unique_ptr<Interface> pimpl;

public:
    template <typename Object, typename Strategy>
    StatefulTE(Object&& object, Strategy&& strategy)
    {
        // object and strategy must be shared_ptr. so if they not already are
        // the following code will convert them (note that this happens at compile time)
        if constexpr (is_shared_ptr<std::remove_cvref_t<Object>>::value)
        {
            if constexpr (is_shared_ptr<std::remove_cvref_t<Strategy>>::value)
            {
                pimpl = std::make_unique<Implementation<typename std::remove_cvref_t<Object>::element_type,
                                                        typename std::remove_cvref_t<Strategy>::element_type>>(
                    std::forward<Object>(object), std::forward<Strategy>(strategy));
            }
            else
            {
                pimpl = std::make_unique<Implementation<typename std::remove_cvref_t<Object>::element_type,
                                                        typename std::remove_cvref_t<Strategy>>>(
                    std::forward<Object>(object),
                    std::make_shared<std::remove_cvref_t<Strategy>>(std::forward<Strategy>(strategy)));
            }
        }
        else
        {
            if constexpr (is_shared_ptr<std::remove_cvref_t<Strategy>>::value)
            {
                pimpl = std::make_unique<Implementation<typename std::remove_cvref_t<Object>,
                                                        typename std::remove_cvref_t<Strategy>::element_type>>(
                    std::make_shared<std::remove_cvref_t<Object>>(std::forward<Object>(object)),
                    std::forward<Strategy>(strategy));
            }
            else
            {
                pimpl = std::make_unique<
                    Implementation<typename std::remove_cvref_t<Object>, typename std::remove_cvref_t<Strategy>>>(
                    std::make_shared<std::remove_cvref_t<Object>>(std::forward<Object>(object)),
                    std::make_shared<std::remove_cvref_t<Strategy>>(std::forward<Strategy>(strategy)));
            }
        }
    }
    void getTreat() { pimpl->getTreat(); }
    void getPetted() { pimpl->getPetted(); }
};


struct Cat
{
    void meow() {}
    void purr() {}
    void scratch() {}
};
struct Dog
{
    void bark() {};
    void sit() {};
    int hairsShedded{0};
    auto hairsSheddedCount() { return hairsShedded; }
    void shed() { hairsShedded += 1'000'000; }
};

struct PetStrategy1
{
    void getTreat(Cat& cat)
    {
        cat.meow();
        cat.scratch();
    }
    void getPetted(Cat& cat) { cat.purr(); }
    void getTreat(Dog& dog) { dog.sit(); }
    void getPetted(Dog& dog)
    {
        dog.bark();
        dog.shed();
    }
};

struct PetStrategy2
{
    int treatsSpent{0};
    auto treatsSpentCount() { return treatsSpent; }
    void getTreat(Cat& cat)
    {
        ++treatsSpent;
        cat.meow();
    }
    void getPetted(Cat& cat)
    {
        cat.purr();
        cat.purr();
    }
    void getTreat(Dog& dog)
    {
        ++treatsSpent;
        dog.sit();
    }
    void getPetted(Dog& dog)
    {
        dog.sit();
        dog.shed();
    }
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
    std::vector<StatefulTE> v;
    v.emplace_back(StatefulTE(rover, s2));
    v.emplace_back(StatefulTE(lazy, s1));

    for (auto&& e : v)
    {
        e.getTreat();
        e.getPetted();
    }
    return 0;
}
