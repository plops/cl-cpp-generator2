#pragma once
#include <concepts>
#include <cstring>
#include <iostream>
#include <type_traits>
template<typename T>
constexpr inline T *c_resource_null_value = nullptr;
// two api schemas for destructors and constructor
// 1) thing* construct();      void destruct(thing*)
// 2) void construct(thing**); void destruct(thing**)
//
// modifiers like replace(..) exist only for schema 2) act like auto
// construct(thing**)
template<typename T, auto *ConstructFunction, auto *DestructFunction>
class c_resource {
public:
    using pointer = T *;
    using const_pointer = std::add_const_t<T> *;
    using element_type = T;

private:
    using Constructor = decltype(ConstructFunction);
    using Destructor = decltype(DestructFunction);

    static constexpr Constructor construct = ConstructFunction;
    static constexpr Destructor destruct = DestructFunction;
    static constexpr T *null = c_resource_null_value<T>;
    struct construct_t {};

public:
    static constexpr construct_t constructed = {};
    [[nodiscard]] constexpr c_resource() noexcept = default;
    [[nodiscard]] constexpr explicit c_resource(construct_t) noexcept
        requires std::is_invocable_r_v<T *, Constructor>
        : ptr_{construct()} {
        std::cout << "construct75" << std::endl;
    };
    template<typename... Ts>
        requires(((0 < sizeof...(Ts)) &&
                  (std::is_invocable_r_v<T *, Constructor, Ts...>) ))
    [[nodiscard]] constexpr explicit(sizeof...(Ts) == 1)
            c_resource(Ts &&...Args) noexcept
        : ptr_{construct(static_cast<Ts &&>(Args)...)} {
        std::cout << "construct83 " << __PRETTY_FUNCTION__ << std::endl;
    };
    template<typename... Ts>
        requires(sizeof...(Ts) > 0 &&
                 requires(T *p, Ts... Args) {
                     { construct(&p, Args...) } -> std::same_as<void>;
                 })
    [[nodiscard]] constexpr explicit(sizeof...(Ts) == 1)
            c_resource(Ts &&...Args) noexcept
        : ptr_{null} {
        construct(&ptr_, static_cast<Ts &&>(Args)...);
        std::cout << "construct93" << std::endl;
    };
    template<typename... Ts>
        requires(std::is_invocable_v<Constructor, T **, Ts...>)
    [[nodiscard]] constexpr auto emplace(Ts &&...Args) noexcept {
        _destruct(ptr_);
        ptr_ = null;

        std::cout << "emplace" << std::endl;
        return construct(&ptr_, static_cast<Ts &&>(Args)...);
    };
    [[nodiscard]] constexpr c_resource(c_resource &&other) noexcept {
        ptr_ = other.ptr_;
        other.ptr_ = null;

        std::cout << "copy104" << std::endl;
    };
    constexpr c_resource &operator=(c_resource &&rhs) noexcept {
        if (!(this == &rhs)) {
            _destruct(ptr_);
            ptr_ = rhs.ptr_;
            rhs.ptr_ = null;

            std::cout << "operator= " << __PRETTY_FUNCTION__ << std::endl;
        }
        return *this;
    };
    constexpr void swap(c_resource &other) noexcept {
        auto ptr = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = ptr;

        std::cout << "swap" << std::endl;
    };
    static constexpr bool destructible =
            ((std::is_invocable_v<Destructor, T *>) |
             (std::is_invocable_v<Destructor, T **>) );

    constexpr ~c_resource() noexcept = delete;
    constexpr ~c_resource() noexcept
        requires destructible
    {
        _destruct(ptr_);
        std::cout << "destruct129 " << __PRETTY_FUNCTION__ << std::endl;
    };
    constexpr void clear() noexcept
        requires destructible
    {
        _destruct(ptr_);
        ptr_ = null;

        std::cout << "clear" << std::endl;
    };
    constexpr c_resource &operator=(std::nullptr_t) noexcept {
        clear();
        std::cout << "operator=137" << std::endl;
        return *this;
    };
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        std::cout << "bool" << std::endl;
        return (ptr_) != (null);
    };
    [[nodiscard]] constexpr bool empty() const noexcept {
        std::cout << "empty" << std::endl;
        return ptr_ == null;
    };
    [[nodiscard]] constexpr friend bool have(const c_resource &r) noexcept {
        std::cout << "have" << std::endl;
        return (r.ptr_) != (null);
    };
    auto operator<=>(const c_resource &) = delete;
    [[nodiscard]] bool operator==(const c_resource &rhs) const noexcept {
        std::cout << "operator==" << std::endl;
        return 0 == std::memcmp(ptr_, rhs.ptr_, sizeof(T));
    };
    // this is the code that clang++ uses (my case)
    [[nodiscard]] constexpr operator pointer() noexcept { return like(*this); };
    [[nodiscard]] constexpr operator const_pointer() const noexcept {
        return like(*this);
    };
    [[nodiscard]] constexpr pointer operator->() noexcept { return like(*this); };
    [[nodiscard]] constexpr const_pointer operator->() const noexcept {
        return like(*this);
    };
    [[nodiscard]] constexpr pointer get() noexcept { return like(*this); };
    [[nodiscard]] constexpr const_pointer get() const noexcept {
        return like(*this);
    };

private:
    static constexpr auto like(c_resource &self) noexcept { return self.ptr_; }
    static constexpr auto like(const c_resource &self) noexcept {
        return static_cast<const_pointer>(self.ptr_);
    }

public:
    constexpr void reset(pointer ptr = null) noexcept {
        _destruct(ptr_);
        ptr_ = ptr;

        std::cout << "reset" << std::endl;
    };
    constexpr pointer release() noexcept {
        auto ptr = ptr_;
        ptr_ = null;

        std::cout << "release" << std::endl;
        return ptr;
    };
    template<auto *CleanupFunction>
    struct guard {
    public:
        using cleaner = decltype(CleanupFunction);

        constexpr guard(c_resource &Obj) noexcept : ptr_{Obj.ptr_} {
            std::cout << "guard" << std::endl;
        };
        constexpr ~guard() noexcept {
            if (!(ptr_ == null)) {
                CleanupFunction(ptr_);
            }
            std::cout << "~guard" << std::endl;
        };

    private:
        pointer ptr_;
    };

private:
    constexpr static void _destruct(pointer &p) noexcept
        requires std::is_invocable_v<Destructor, T *>
    {
        if (!(p == null)) {
            std::cout << "_destruct224 T*" << std::endl;
            destruct(p);
        }
    };
    constexpr static void _destruct(pointer &p) noexcept
        requires std::is_invocable_v<Destructor, T **>
    {
        if (!(p == null)) {
            std::cout << "_destruct230 T**" << std::endl;
            destruct(&p);
        }
    };
    pointer ptr_ = null;
};
