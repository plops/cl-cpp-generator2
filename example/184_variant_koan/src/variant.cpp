//
// Created by martin on 5/16/25.
//

#include <iostream>
#include <variant>
#include <vector>

struct PhilosophersElixir {
    int                  potency;
    std::string          effect;
    friend std::ostream& operator<<(std::ostream& os, const PhilosophersElixir& ph) {
        os << ph.potency << " " << ph.effect << " " << ph.effect;
        return os;
    }
};

using AlchemistsStone = std::variant<int, std::string, PhilosophersElixir>;

void describe_stone(const AlchemistsStone& stone) {
    // std::visit([](auto&& arg) {
    //     using T = std::decay_t<decltype(arg)>;
    //     if constexpr (std::is_same_v<T, PhilosophersElixir>) {
    //         std::cout << arg.potency << " " << arg.effect << std::endl;
    //     } else if constexpr (std::is_same_v<T, int>) {
    //         std::cout << arg << std::endl;
    //     } else if constexpr (std::is_same_v<T, std::string>) {
    //         std::cout << arg << std::endl;
    //     }
    // }, stone);
    // std::visit([](int val) { std::cout << val << '\n'; }, [](const std::string& val) { std::cout << val << '\n'; },
    //            [](const PhilosophersElixir& p) { std::cout << p.potency << '\n'; }, stone);

    struct Visitor {
        void operator()(const PhilosophersElixir& x) { std::cout << x.potency << " " << x.effect << "\n"; }
        void operator()(int x) { std::cout << x << "\n"; }
        void operator()(const std::string& x) { std::cout << x << "\n"; }
    };
    std::visit(Visitor{}, stone);
}

int main() {
    std::vector<AlchemistsStone> observations;
    observations.push_back(42);
    observations.push_back("Abracadabra");
    observations.push_back(PhilosophersElixir{100, "Clarity of Mind"});
    observations.push_back(7);

    std::cout << "Varius's Observations:" << std::endl;
    for (const auto& stone_form : observations) { describe_stone(stone_form); }
    return 0;
}
