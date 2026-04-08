export module ReflectModule;
import <iostream>;
import <meta>; // Proposed header for reflection

export struct User {
    std::string_view name;
    int age;
};

export template <typename T>
void print_fields(const T& obj) {
    // ^T is the reflection operator
    // [:field:] is the "splicer" to access the member
    template for (constexpr auto field : std::meta::nonstatic_data_members_of(^T)) {
        std::cout << std::meta::identifier_of(field) << ": " 
                  << obj.[:field:] << "\n";
    }
}