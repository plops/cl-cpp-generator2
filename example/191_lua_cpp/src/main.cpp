#include <sol/sol.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>

enum class BlendMode {
    Normal,
    Add,
    Multiply
};

struct Colour {
    float r{0.0f};
    float g{0.0f};
    float b{0.0f};

    Colour() = default;
    Colour(float r_, float g_, float b_) : r{r_}, g{g_}, b{b_} {}
};

struct Image {
    int width{0};
    int height{0};
    std::vector<Colour> pixels;

    Image(int w, int h)
        : width{w}, height{h}, pixels(static_cast<std::size_t>(w) * h) {}

    void clear(const Colour& c) {
        std::ranges::fill(pixels, c);
    }

    void set_pixel(int x, int y, const Colour& c) {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return; // ignore out of bounds
        }
        pixels[static_cast<std::size_t>(y) * width + x] = c;
    }

    [[nodiscard]] Colour get_pixel(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return {};
        }
        return pixels[static_cast<std::size_t>(y) * width + x];
    }

    void save_ppm(const std::filesystem::path& path,
                  BlendMode mode = BlendMode::Normal) const
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file: " + path.string());
        }

        out << "P6\n" << width << ' ' << height << "\n255\n";

        auto clamp_byte = [](float v) {
            v = std::clamp(v, 0.0f, 1.0f);
            return static_cast<std::uint8_t>(std::lround(v * 255.0f));
        };

        float factor = 1.0f;
        switch (mode) {
            case BlendMode::Normal:   factor = 1.0f; break;
            case BlendMode::Add:      factor = 1.2f; break;   // just brighten a bit
            case BlendMode::Multiply: factor = 0.8f; break;   // just darken a bit
        }

        for (const auto& p : pixels) {
            std::uint8_t rgb[3] = {
                clamp_byte(p.r * factor),
                clamp_byte(p.g * factor),
                clamp_byte(p.b * factor)
            };
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
};

// Simple C++20 UTF-8 caption holder.
struct App {
    std::u8string caption_ = u8"Lua Image Demo — こんにちは";

    [[nodiscard]] std::string caption() const {
        // Narrow copy; assumes caption_ is valid UTF-8 (Lua strings are UTF-8)
        return std::string{caption_.begin(), caption_.end()};
    }

    void set_caption(std::string_view utf8) {
        caption_.assign(
            reinterpret_cast<const char8_t*>(utf8.data()),
            reinterpret_cast<const char8_t*>(utf8.data() + utf8.size())
        );
    }
};

// Convenience type to show exposing std::vector
using IntVector = std::vector<int>;

int main() {
    sol::state lua;
    lua.open_libraries(sol::lib::base,
                       sol::lib::math,
                       sol::lib::string,
                       sol::lib::table);

    // Expose enum
    lua.new_enum("BlendMode",
        "Normal",   BlendMode::Normal,
        "Add",      BlendMode::Add,
        "Multiply", BlendMode::Multiply
    );

    // Expose Colour
    lua.new_usertype<Colour>("Colour",
        sol::constructors<Colour(), Colour(float, float, float)>(),
        "r", &Colour::r,
        "g", &Colour::g,
        "b", &Colour::b
    );

    // Expose Image
    lua.new_usertype<Image>("Image",
        sol::constructors<Image(int, int)>(),
        "width",       &Image::width,
        "height",      &Image::height,
        "clear",       &Image::clear,
        "set_pixel",   &Image::set_pixel,
        "get_pixel",   &Image::get_pixel,
        "save",        &Image::save_ppm
    );

    // Expose std::vector<int> as a "class" in Lua
    lua.new_usertype<IntVector>("IntVector",
        sol::constructors<>() /* default is fine, but we construct from C++ */,
        "size", [](const IntVector& self) { return self.size(); },
        "get",  [](const IntVector& self, std::size_t i) { return self.at(i); },
        "set",  [](IntVector& self, std::size_t i, int v) { self.at(i) = v; }
    );

    lua.set_function("make_vector", []() {
        // Just a Fibonacci-ish example
        return IntVector{1, 1, 2, 3, 5, 8, 13};
    });

    // Expose App for UTF-8 strings
    App app;

    lua["app"] = &app;

    lua.set_function("get_caption", [&app]() {
        return app.caption();          // std::string (UTF-8)
    });

    lua.set_function("set_caption", [&app](std::string_view s) {
        app.set_caption(s);            // stored as std::u8string
    });

    try {
        // Run Lua script
        lua.script_file("scripts/demo.lua");
    } catch (const sol::error& e) {
        std::cerr << "Lua error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "C++ error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Final caption from C++: " << app.caption() << '\n';
    return 0;
}


