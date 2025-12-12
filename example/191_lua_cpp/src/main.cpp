#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "linenoise.h"

enum class BlendMode { Normal, Add, Multiply };

struct Colour {
    float r{0.0f};
    float g{0.0f};
    float b{0.0f};

    Colour() = default;
    Colour(float r_, float g_, float b_) : r{r_}, g{g_}, b{b_} {}
};

struct Image {
    int                 width{0};
    int                 height{0};
    std::vector<Colour> pixels;

    Image(int w, int h) :
        width{w}, height{h}, pixels(static_cast<std::size_t>(w) * static_cast<std::size_t>(h)) // <- both cast
    {}

private:
    [[nodiscard]] std::size_t index(int x, int y) const noexcept {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    }

public:
    void clear(const Colour& c) { std::ranges::fill(pixels, c); }

    void set_pixel(int x, int y, const Colour& c) {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return; // ignore out of bounds
        }
        pixels[index(x, y)] = c;
    }

    [[nodiscard]] Colour get_pixel(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) { return {}; }
        return pixels[index(x, y)];
    }

    void save_ppm(const std::filesystem::path& path, BlendMode mode = BlendMode::Normal) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) { throw std::runtime_error("Cannot open output file: " + path.string()); }

        out << "P6\n" << width << ' ' << height << "\n255\n";

        auto clamp_byte = [](float v) {
            v = std::clamp(v, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(std::lround(v * 255.0f));
        };

        float factor = 1.0f;
        switch (mode) {
        case BlendMode::Normal:
            factor = 1.0f;
            break;
        case BlendMode::Add:
            factor = 1.2f;
            break;
        case BlendMode::Multiply:
            factor = 0.8f;
            break;
        }

        for (const auto& p : pixels) {
            std::uint8_t rgb[3] = {clamp_byte(p.r * factor), clamp_byte(p.g * factor), clamp_byte(p.b * factor)};
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
};

// Simple UTF-8 app caption holder
struct App {
    std::u8string caption_ = u8"Lua Image Demo — こんにちは";

    [[nodiscard]] std::string caption() const { return std::string{caption_.begin(), caption_.end()}; }

    void set_caption(std::string_view utf8) {
        caption_.assign(reinterpret_cast<const char8_t*>(utf8.data()),
                        reinterpret_cast<const char8_t*>(utf8.data() + utf8.size()));
    }
};

// Demonstrate exposing std::vector<int>
using IntVector = std::vector<int>;
namespace {
    sol::state* g_lua_for_completion = nullptr;

    void lua_completion_callback(const char* buf, linenoiseCompletions* lc) {
        if (!g_lua_for_completion || !buf) {
            return;
        }

        std::string_view line(buf);
        const auto pos = line.find_last_of(" \t");
        std::string_view word = (pos == std::string_view::npos) ? line : line.substr(pos + 1);

        auto add_if_matches = [&](std::string_view candidate) {
            if (word.empty() || candidate.rfind(word, 0) == 0) { // starts_with
                std::string s(candidate);
                linenoiseAddCompletion(lc, s.c_str());
            }
        };

        // Lua keywords
        static constexpr std::string_view keywords[] = {
            "and", "break", "do", "else", "elseif", "end", "false", "for",
            "function", "goto", "if", "in", "local", "nil", "not", "or",
            "repeat", "return", "then", "true", "until", "while"
        };

        for (auto kw : keywords) {
            add_if_matches(kw);
        }

        // Globals from current Lua state
        sol::table globals = g_lua_for_completion->globals();
        for (auto& kv : globals) {
            sol::object key = kv.first;
            if (key.is<std::string>()) {
                std::string name = key.as<std::string>();
                add_if_matches(name);
            }
        }
    }

    void run_repl(sol::state& lua) {
        g_lua_for_completion = &lua;
        linenoiseSetCompletionCallback(lua_completion_callback);
        linenoiseHistoryLoad("lua_history.txt");

        while (true) {
            char* line = linenoise("lua> ");
            if (line == nullptr) { // Ctrl-D / EOF
                break;
            }

            std::string input(line);
            linenoiseFree(line);

            if (input.empty()) {
                continue;
            }

            if (input == "quit" || input == "exit") {
                break;
            }

            sol::load_result chunk = lua.load(input, "repl");
            if (!chunk.valid()) {
                sol::error err = chunk;
                std::cerr << "Compile error: " << err.what() << '\n';
                continue;
            }

            sol::protected_function func = chunk;
            sol::protected_function_result result = func();
            if (!result.valid()) {
                sol::error err = result;
                std::cerr << "Runtime error: " << err.what() << '\n';
            }

            linenoiseHistoryAdd(input.c_str());
            linenoiseHistorySave("lua_history.txt");
        }

        g_lua_for_completion = nullptr;
    }
}
int main() {
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::string, sol::lib::table);

    // --- Enum (sol3 style) ------------------------------------
    lua.new_enum("BlendMode", "Normal", BlendMode::Normal, "Add", BlendMode::Add, "Multiply", BlendMode::Multiply);

    // --- Colour usertype --------------------------------------
    lua.new_usertype<Colour>("Colour", sol::constructors<Colour(), Colour(float, float, float)>(), "r", &Colour::r, "g",
                             &Colour::g, "b", &Colour::b);

    // --- Image usertype ---------------------------------------
    lua.new_usertype<Image>("Image", sol::constructors<Image(int, int)>(), "width", &Image::width, "height",
                            &Image::height, "clear", &Image::clear, "set_pixel", &Image::set_pixel, "get_pixel",
                            &Image::get_pixel,
                            // wrap save_ppm so Lua can pass a string; sol3 docs style
                            "save", [](const Image& self, const std::string& filename, BlendMode mode) {
                                self.save_ppm(std::filesystem::path(filename), mode);
                            });

    // --- std::vector<int> exposure ----------------------------
    lua.new_usertype<IntVector>(
            "IntVector", sol::constructors<>(), "size", [](const IntVector& self) { return self.size(); }, "get",
            [](const IntVector& self, std::size_t i) { return self.at(i); }, "set",
            [](IntVector& self, std::size_t i, int v) { self.at(i) = v; });

    lua.set_function("make_vector", []() {
        // Fibonacci-like example
        return IntVector{1, 1, 2, 3, 5, 8, 13};
    });

    // --- UTF-8 App binding ------------------------------------
    App app;

    // pointer / reference semantics as in the docs' Doge example
    lua["app"] = &app;

    lua.set_function("get_caption", [&app]() { return app.caption(); });

    lua.set_function("set_caption", [&app](std::string_view s) { app.set_caption(s); });

    try {
        lua.script_file("scripts/demo.lua");
    }
    catch (const sol::error& e) {
        std::cerr << "Lua error: " << e.what() << '\n';
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "C++ error: " << e.what() << '\n';
        return 1;
    }

    // Interactive Lua REPL (after running demo.lua)
    run_repl(lua);

    std::cout << "Final caption from C++: " << app.caption() << '\n';
    return 0;
}


// lua> img = Image.new(64, 64)
// lua> set_caption("fiens")
// lua> print(get_caption())