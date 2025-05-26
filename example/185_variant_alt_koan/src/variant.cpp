#include <iostream>
#include <memory>
#include <string>
#include <type_traits> // For std::is_same_v
#include <variant>
#include <vector>
using namespace std;

class Entry {
    public:
    explicit Entry(const string& name) : name_{move(name)}
    // ~Entry() = default;
    string getName() const {return name_;}
private:
    string name_;
};

template<class T>
class TEntry : public Entry {
public:
    TEntry():Entry(stringify(T)){}
};

using IntEntry = TEntry<int>;
using StringEntry = TEntry<string>;
using VIntEntry = TEntry<vector<int>>;

using ScribesLedgerEntry = variant<IntEntry, StringEntry, VIntEntry>;
//
// using TypeOfDetailedDescription = variant_alternative_t<1, ScribesLedgerEntry>;
//
// static_assert(is_same_v<TypeOfDetailedDescription, string>,
//               "The Scribe's detailed description is not what you thought.");

#define stringify(x) (#x)

template<int I=0>
shared_ptr<Entry> frob(string name) {
    if constexpr (I < variant_size_v<ScribesLedgerEntry>) {
        using CurrentType = variant_alternative_t<I, ScribesLedgerEntry>;
        cout << stringify(CurrentType) << endl;
        if (name == stringify(CurrentType)) {
            return make_shared<CurrentType>();
        }
        return frob<I+1>(name);
    }
    return nullptr;
}


int main() {
    frob("IntEntry");
    return 0;
}
