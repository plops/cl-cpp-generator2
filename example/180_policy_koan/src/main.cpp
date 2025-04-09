#include <iostream>
using namespace std;
template <class T>
class OpNewCreator {
public:
    T* create() {
        std::cout << "OpNewCreator<T>::create " << " sizeof(T)='" << sizeof(T) << "' " << " sizeof(*this)='"
                  << sizeof(*this) << "' " << " sizeof(decltype(*this))='" << sizeof(decltype(*this)) << "' "
                  << " sizeof(OpNewCreator<T>)='" << sizeof(OpNewCreator<T>) << "' " << std::endl;
        return new T;
    }

protected:
    ~OpNewCreator() {};
};
template <class T>
class MallocCreator {
public:
    T* create() {
        std::cout << "MallocCreator<T>::create " << " sizeof(T)='" << sizeof(T) << "' " << " sizeof(*this)='"
                  << sizeof(*this) << "' " << " sizeof(decltype(*this))='" << sizeof(decltype(*this)) << "' "
                  << " sizeof(MallocCreator<T>)='" << sizeof(MallocCreator<T>) << "' " << std::endl;
        auto buf{malloc(sizeof(T))};
        if (!buf) { return nullptr; }
        return new (buf) T;
    }

protected:
    ~MallocCreator() {};
};
template <class T>
class PrototypeCreator {
public:
    T* create() {
        std::cout << "PrototypeCreator<T>::create " << " sizeof(T)='" << sizeof(T) << "' " << " sizeof(*this)='"
                  << sizeof(*this) << "' " << " sizeof(decltype(*this))='" << sizeof(decltype(*this)) << "' "
                  << " sizeof(PrototypeCreator<T>)='" << sizeof(PrototypeCreator<T>) << "' " << std::endl;
        return prototype ? prototype->clone() : nullptr;
    }
    explicit PrototypeCreator(T* obj = nullptr) : prototype{obj} {}
    T*   getPrototype() { return prototype; }
    void setPrototype(T* obj) { prototype = obj; }

private:
    T* prototype;

protected:
    ~PrototypeCreator() {};
};
class Widget {
    int             a;
    float           f;
    array<char, 20> c;

public:
    Widget* clone() { return new Widget; }
};
template <template <class Created> class CreationPolicy = OpNewCreator>
class WidgetManager : public CreationPolicy<Widget> {
public:
    WidgetManager() {}
    void switchPrototype(Widget* newPrototype) {
        CreationPolicy<Widget>& myPolicy = *this;
        delete (myPolicy.getPrototype());
        myPolicy.setPrototype(newPrototype);
    }
};
using MyWidgetMgr = WidgetManager<OpNewCreator>;

int main(int argc, char** argv) {
    auto wm0{MyWidgetMgr()};
    auto e0{wm0.create()};
    auto wm1{WidgetManager<MallocCreator>()};
    auto e1{wm1.create()};
    auto wm2{WidgetManager<PrototypeCreator>()};
    wm2.setPrototype(e1);
    auto e2{wm2.create()};
    wm2.switchPrototype(e2);
    std::cout << "" << " sizeof(wm0)='" << sizeof(wm0) << "' " << " sizeof(wm1)='" << sizeof(wm1) << "' "
              << " sizeof(wm2)='" << sizeof(wm2) << "' " << std::endl;
    return 0;
}
