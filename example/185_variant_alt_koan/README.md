^Okay, here is a koan for `std::variant_alternative`.

**The Koan: The Scribe's Ledger**

```cpp
#include <variant>
#include <string>
#include <vector>
#include <type_traits> // For std::is_same_v
#include <iostream>    // For the success message

// The ancient Scribe maintains a ledger with three types of entries:
// 0: A simple count (int)
// 1: A detailed description (std::string)
// 2: A list of witness IDs (std::vector<int>)
using ScribesLedgerEntry = std::variant<int, std::string, std::vector<int>>;

// The Scribe needs to know, *before writing any entry*,
// what specific C++ type corresponds to the "detailed description" entry.
// This knowledge is needed at compile time, for temple rituals (template metaprogramming).

// Meditate on this:
// What is the type of the alternative at index 1 in ScribesLedgerEntry?
// Fill in the __BLANK__ to reveal its true nature.

using TypeOfDetailedDescription = /* __BLANK__ */;

// If your understanding is true, this ancient spell (static_assert) will hold.
static_assert(std::is_same_v<TypeOfDetailedDescription, std::string>,
              "The Scribe's detailed description is not what you thought.");

// To complete your meditation, compile and run this.
// No output means failure (or the static_assert failed to compile).
// A success message means enlightenment.

int main() {
    // If the static_assert passed, this line will be reached.
    std::cout << "Enlightenment achieved: The Scribe's detailed description is indeed a std::string." << std::endl;

    // Further contemplation (optional):
    // What if the Scribe wanted to declare a variable of this type?
    TypeOfDetailedDescription currentDescription = "A dragon was sighted near the western gate.";
    std::cout << "An example description: " << currentDescription << std::endl;

    // What is the type of the "list of witness IDs" (index 2)?
    using TypeOfWitnessList = std::variant_alternative_t<2, ScribesLedgerEntry>;
    static_assert(std::is_same_v<TypeOfWitnessList, std::vector<int>>, "Witness list type mismatch!");
    TypeOfWitnessList witnesses = {101, 102, 105};
    std::cout << "Witnesses: ";
    for(int id : witnesses) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Expected output:
Enlightenment achieved: The Scribe's detailed description is indeed a std::string.
An example description: A dragon was sighted near the western gate.
Witnesses: 101 102 105
*/
```

**The Question:**

What should replace `/* __BLANK__ */` to make the `static_assert` pass and reveal the type of the "detailed description" entry?

---

**The Solution:**

The `__BLANK__` should be filled with:

```cpp
std::variant_alternative_t<1, ScribesLedgerEntry>
```

**Explanation:**

*   `std::variant_alternative_t` is a type alias template.
*   It takes two template arguments:
    1.  `std::size_t I`: The 0-based index of the alternative whose type you want to query. In our koan, "detailed description" is the second type listed in `ScribesLedgerEntry`, so its index is `1`.
    2.  `class Variant`: The `std::variant` type itself. In our koan, this is `ScribesLedgerEntry`.

So, `std::variant_alternative_t<1, ScribesLedgerEntry>` resolves, at compile time, to the type `std::string`. The `static_assert` then checks if this resolved type is indeed the same as `std::string`, which it is, allowing the program to compile and run.

This demonstrates that `std::variant_alternative` (and its `_t` helper) allows you to introspect the types contained within a `std::variant` purely at compile time, without needing an instance of the variant or knowing its current state. This is invaluable for generic programming and metaprogramming.