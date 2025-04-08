#include <iostream>
using namespace std;
template <class T>
class OpNewCreator {
public:
    T* create() { return new T; }
};
template <class T>
class MallocCreator {
public:
    T* create() {
        auto buf{malloc(sizeof(T))};
        if (!buf) { return nullptr; }
        return new (buf) T;
    }
};
template <class T>
class PrototypeCreator {
public:
    T* create() { return prototype ? prototype->clone() : nullptr; }
    explicit PrototypeCreator(T* obj) : prototype{obj} {}
    T*   getPrototype() { return prototype; }
    void setPrototype(T* obj) { prototype = obj; }

private:
    T* prototype;
};
class Widget {
    int   a;
    float f;
};
template <template <class Created> class CreationPolicy>
class WidgetManager : public CreationPolicy<Widget> {};
using MyWidgetMgr = WidgetManager<OpNewCreator>;

int main(int argc, char** argv) {
    auto wm{MyWidgetMgr()};
    return 0;
}
