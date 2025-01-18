#include <memory>
#include <vector>
#include <iostream>
#include <string_view>
#include <string>
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
    std::string name_;
    Cat(std::string_view name): name_(name) {}
    void meow() { std::cout << name_ << " meow" << std::endl; }
    void purr() { std::cout << name_ << " purr" << std::endl; }
    void scratch() { std::cout << name_ << " scratch" << std::endl; }
};
struct Dog
{
    std::string name_;
    Dog(std::string_view name): name_(name) {}
    void bark() { std::cout << name_ << " bark" << std::endl; };
    void sit() { std::cout << name_ << " sit" << std::endl; };
    int hairsShedded{0};
    auto hairsSheddedCount() { return hairsShedded; }
    void shed()
    {
        hairsShedded += 1'000'000;
        std::cout << name_ << " shed " << hairsShedded << std::endl;
    }
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
    //auto rover{Dog("rover")};
    auto rover{std::make_shared<Dog>("rover")};
    auto lazy{Cat("lazy")};
    auto s1{PetStrategy1()};
    auto s2{std::make_shared<PetStrategy2>()};
    std::vector<StatefulTE> v;
    v.emplace_back(StatefulTE(rover, s1));
    v.emplace_back(StatefulTE(lazy, s1));
    v.emplace_back(StatefulTE(rover, s2));
    v.emplace_back(StatefulTE(lazy, s2));
    for (auto&& e : v)
    {
        e.getTreat();
        e.getPetted();
        std::cout << "rover hairs shed: " <<  rover->hairsSheddedCount() << std::endl;
        std::cout << "s2 treats: " << s2->treatsSpentCount() << std::endl;
    }
    return 0;
}
