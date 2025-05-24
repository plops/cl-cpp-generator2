#include <iostream>
#include <string>
#include <type_traits> // For std::is_same_v
#include <variant>
#include <vector>

using namespace std;
using ScribesLedgerEntry = variant<int, string, vector<int>>;

using TypeOfDetailedDescription = variant_alternative_t<1, ScribesLedgerEntry>;

static_assert(is_same_v<TypeOfDetailedDescription, string>,
              "The Scribe's detailed description is not what you thought.");

int main() {
    TypeOfDetailedDescription description="A dragon was sighted near the western gate.";
    cout << description << endl;

    using TypeOfWitnessList = variant_alternative_t<2, ScribesLedgerEntry>;
    TypeOfWitnessList witnessList={101,102,105};
    for (auto id : witnessList) {
        cout << id << endl;
    }
    return 0;
}
