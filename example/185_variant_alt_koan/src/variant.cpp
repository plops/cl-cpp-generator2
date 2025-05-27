#include <iostream>
#include <memory>
#include <string>
#include <type_traits> // For std::is_same_v
#include <variant>
#include <vector>
using namespace std;

#define stringify(x) (#x)


class Entry {
public:
    // Constructor now takes an alias_name (e.g., "IntEntry" for TEntry<int>)
    Entry(const string& alias_name) :
        alias_name_{alias_name} {}

    // Returns the alias name (e.g., "IntEntry"), used for lookup.
    string getName() const { return alias_name_; }

    // Specific getters if needed
    string getAliasName() const { return alias_name_; }

private:
    string alias_name_;         // e.g., "IntEntry"
};

template <class T>
class TEntry : public Entry {
public:
    using type = T; // Expose T, useful for type assertions or introspection

    // Constructor takes the alias_name (e.g., "IntEntry").
    // It then constructs the base Entry, passing this alias_name
    // and stringify(T) as the internal_type_name.
    explicit TEntry(const string& alias_name) : Entry(alias_name) {}
};

using IntEntry    = TEntry<int>;
using StringEntry = TEntry<string>;
using VIntEntry   = TEntry<vector<int>>;

// For T == vector<int>, stringify(T) will result in "vector<int>"
// For T == string (due to 'using namespace std;'), stringify(T) will be "string"

using ScribesLedgerEntry = variant<IntEntry, StringEntry, VIntEntry>;

// Statically known names for the types in ScribesLedgerEntry.
// The stringify macro works correctly on type aliases here (outside template context).
static const char* ScribesLedgerEntryNames[] = {
        stringify(IntEntry),    // Results in "IntEntry"
        stringify(StringEntry), // Results in "StringEntry"
        stringify(VIntEntry)    // Results in "VIntEntry"
};

// Compile-time check to ensure the names array matches the variant's size.
static_assert(sizeof(ScribesLedgerEntryNames) / sizeof(ScribesLedgerEntryNames[0]) ==
                      variant_size_v<ScribesLedgerEntry>,
              "Mismatch between ScribesLedgerEntryNames array size and ScribesLedgerEntry variant size.");

// Factory function 'frob'
// Iterates through variant types at compile time to find a match for 'name_to_find'.
template <int I = 0>
shared_ptr<Entry> frob(const string& name_to_find) { // Pass string by const reference
    if constexpr (I < variant_size_v<ScribesLedgerEntry>) {
        // Get the I-th type in the variant (e.g., IntEntry)
        using CurrentType = variant_alternative_t<I, ScribesLedgerEntry>;

        // Debug output: show which type string we are checking against
        cout << "frob: Checking if '" << name_to_find << "' matches '" << ScribesLedgerEntryNames[I] << "'" << endl;

        if (name_to_find == ScribesLedgerEntryNames[I]) {
            // Found a match. Create an instance of CurrentType.
            // CurrentType (e.g., IntEntry) is TEntry<SomeType> (e.g., TEntry<int>).
            // Its constructor TEntry(const string& alias_name) needs the alias name.
            return make_shared<CurrentType>(ScribesLedgerEntryNames[I]);
        }
        // No match, recurse for the next type in the variant.
        return frob<I + 1>(name_to_find);
    }
    // Base case for recursion: all types checked, no match found.
    return nullptr;
}

int main() {
    shared_ptr<Entry> entry;

    cout << "--- Testing IntEntry ---" << endl;
    entry = frob("IntEntry");
    if (entry) {
        cout << "Successfully created entry." << endl;
        cout << "  getName(): " << entry->getName() << endl;                         // Expected: "IntEntry"
        cout << "  getAliasName(): " << entry->getAliasName() << endl;               // Expected: "IntEntry"
    }
    else { cout << "Failed to create entry for 'IntEntry'." << endl; }
    cout << endl;

    cout << "--- Testing StringEntry ---" << endl;
    entry = frob("StringEntry");
    if (entry) {
        cout << "Successfully created entry." << endl;
        cout << "  getName(): " << entry->getName() << endl;                         // Expected: "StringEntry"
        cout << "  getAliasName(): " << entry->getAliasName() << endl;               // Expected: "StringEntry"
    }
    else { cout << "Failed to create entry for 'StringEntry'." << endl; }
    cout << endl;

    cout << "--- Testing VIntEntry ---" << endl;
    entry = frob("VIntEntry");
    if (entry) {
        cout << "Successfully created entry." << endl;
        cout << "  getName(): " << entry->getName() << endl;                         // Expected: "VIntEntry"
        cout << "  getAliasName(): " << entry->getAliasName() << endl;               // Expected: "VIntEntry"
    }
    else { cout << "Failed to create entry for 'VIntEntry'." << endl; }
    cout << endl;

    cout << "--- Testing UnknownEntry ---" << endl;
    entry = frob("UnknownEntry");
    if (entry) { cout << "Unexpectedly created entry for 'UnknownEntry'!" << endl; }
    else { cout << "Correctly failed to create entry for 'UnknownEntry'." << endl; }
    cout << endl;

    // Original static_asserts were checking if variant_alternative_t<1, ScribesLedgerEntry> (which is StringEntry)
    // is std::string. This is false. StringEntry is TEntry<string>.
    // If the intention was to check the underlying type T of StringEntry:
    using TypeForStringEntryCheck = variant_alternative_t<1, ScribesLedgerEntry>; // This is StringEntry

    static_assert(is_same_v<TypeForStringEntryCheck, StringEntry>, "Variant alternative 1 should be StringEntry.");

    // Check the ::type member we added to TEntry (which is T)
    static_assert(is_same_v<StringEntry::type, string>, // or std::string if not `using namespace std;`
                  "StringEntry::type (the T in TEntry<T>) should be std::string.");
    cout << "Static assertions passed." << endl;

    return 0;
}
