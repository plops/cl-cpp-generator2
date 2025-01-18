#include <iostream>
#include <memory>
#include <vector>


class UniversalTE
{
private:
    struct Interface
    {
        virtual ~Interface() = default;
        virtual void getTreat() = 0;
        virtual void getPetted() = 0;
    };


    std::unique_ptr<Interface> pimpl;


    template <typename Type>
    struct is_shared_ptr : std::false_type
    {
    };
    template <typename Type>
    struct is_shared_ptr<std::shared_ptr<Type>> : std::true_type
    {
    };


    // Object and Strategy get copied into the templated Implementation class (so we can't keep track of states in these
    // objects)
    template <typename Object, typename Strategy>
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


        template <typename Object2, typename Strategy2>
        Implementation(Object2&& o, Strategy2&& s) :
            object(std::forward<Object2>(o)), strategy(std::forward<Strategy2>(s))
        {
        }
        void getTreat() override { strategy_().getTreat(object_()); }
        void getPetted() override { strategy_().getPetted(object_()); }
    };


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
    void meow() {}
    void purr() {}
    void scratch() {}
};
struct Dog
{
    void bark() {};
    void sit() {};
    int hairsShedded{0};
    [[nodiscard]] auto hairsSheddedCount() const { return hairsShedded; }
    void shed() { hairsShedded += 1'000'000; }
};

struct PetStrategy1
{
    static void getTreat(Cat& cat)
    {
        cat.meow();
        cat.scratch();
    }
    static void getPetted(Cat& cat) { cat.purr(); }
    static void getTreat(Dog& dog) { dog.sit(); }
    static void getPetted(Dog& dog)
    {
        dog.bark();
        dog.shed();
    }
};

struct PetStrategy2
{
    int treatsSpent{0};
    [[nodiscard]] auto treatsSpentCount() const { return treatsSpent; }
    void getTreat(Cat& cat)
    {
        ++treatsSpent;
        cat.meow();
    }
    static void getPetted(Cat& cat)
    {
        cat.purr();
        cat.purr();
    }
    void getTreat(Dog& dog)
    {
        ++treatsSpent;
        dog.sit();
    }
    static void getPetted(Dog& dog)
    {
        dog.sit();
        dog.shed();
    }
};

int main()
{
    auto rover{std::make_shared<Dog>()};
    auto bella{Dog()};
    auto lazy{Cat()};
    auto s1{PetStrategy1()};
    auto s2{std::make_shared<PetStrategy2>()};

    std::vector<UniversalTE> v;
    v.emplace_back(rover, s2);
    v.emplace_back(lazy, s1);
    v.emplace_back(bella, s2);

    for (auto&& e : v)
    {
        e.getTreat();
        e.getPetted();
        std::cout << "s2 treats: " << s2->treatsSpentCount() << std::endl;
        std::cout << "rover shed: " << rover->hairsSheddedCount() << std::endl;
    }
    return 0;
}
