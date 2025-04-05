//
// Created by martin on 4/1/25.
//

#include <iostream>
#include <memory>
#include <string>
#include <cassert>
#include <vector>

using namespace std;

class Widget {
public:
    Widget(int i, string n) :
        id{i}, name{move(n)} { cout << "Widget " << id << " (" << name << ") created." << endl; }

    virtual      ~Widget() { cout << "Widget " << id << " (" << name << ") destroyed." << endl; }
    virtual void display() const { cout << "Widget ID: " << id << ", Name: " << name << endl; }
    int          getId() const { return id; }

private:
    int    id;
    string name;
};

template <typename T>
class AbstractPtr {
public:
    virtual      ~AbstractPtr() = default;
    virtual bool isValid() const =0;
    virtual T*   get() const = 0;
    virtual T*   operator->() const = 0;
    virtual T&   operator*() const =0;
    AbstractPtr(const AbstractPtr&)            = delete;
    AbstractPtr& operator=(const AbstractPtr&) = delete;
    AbstractPtr(AbstractPtr&&)                 = delete;
    AbstractPtr& operator=(AbstractPtr&&)      = delete;

protected:
    // Constructor for base class use only
    AbstractPtr() = default;
};

template <typename T>
class UniquePtrWrapper : public AbstractPtr<T> {
public:
    explicit UniquePtrWrapper(unique_ptr<T> p):
        ptr_{move(p)} {}

    bool isValid() const override { return static_cast<bool>(ptr_); }
    T*   get() const override { return ptr_.get(); }

    T* operator->() const override {
        assert(isValid() && "Attempting to access through null UniquePtrWrapper");
        return ptr_.get();
    }

    T& operator*() const override {
        assert(isValid() && "Attempting to dereference null UniquePtrWrapper");
        return *ptr_;
    }

    ~UniquePtrWrapper() override = default;

private:
    unique_ptr<T> ptr_;
};

template <typename T>
class SharedPtrWrapper : public AbstractPtr<T> {
public:
    explicit SharedPtrWrapper(shared_ptr<T> p) :
        ptr_{move(p)} {}

    bool isValid() const override { return static_cast<bool>(ptr_); }
    T*   get() const override { return ptr_.get(); }

    T* operator->() const override {
        assert(isValid() && "Attempting to access through null SharedPtrWrapper");
        return ptr_.get();
    }

    T& operator*() const override {
        assert(isValid() && "Attempting to dereference null SharedPtrWrapper");
        return *get();
    }

    ~SharedPtrWrapper() override = default;

private:
    std::shared_ptr<T> ptr_;
};

void processWidget(const AbstractPtr<Widget>& abstractWidgetPtr) {
    cout << "Processing via AbstractPtr: ";
    if (abstractWidgetPtr.isValid()) {
        abstractWidgetPtr->display();
        cout << "(Raw pointer: " << abstractWidgetPtr.get() << ")" << endl;
        Widget& w = *abstractWidgetPtr;
        cout << "Accessed via *: Widget ID " << w.getId() << endl;
    }
    else { cout << "Pointer is null" << endl; }
}

int main(int argc, char* argv[]) {
    cout << "Koan start" << endl;

    auto                     uniqueWidget = make_unique<Widget>(1, "Gizmo");
    UniquePtrWrapper<Widget> uniqueWrapper(move(uniqueWidget));

    processWidget(uniqueWrapper);

    auto                     sharedWidget = make_shared<Widget>(2, "Gadget");
    SharedPtrWrapper<Widget> sharedWrapper(sharedWidget);

    processWidget(sharedWrapper);

    UniquePtrWrapper<Widget> nullWrapper(nullptr);

    processWidget(nullWrapper);

    // Demonstrate polymorphism with a vector

    vector<unique_ptr<AbstractPtr<Widget>>> widgetPointers;

    widgetPointers.push_back(make_unique<UniquePtrWrapper<Widget>>(make_unique<Widget>(3, "Thingamajig")));
    widgetPointers.push_back(make_unique<SharedPtrWrapper<Widget>>(make_shared<Widget>(4, "Doodad")));
    for (const auto& ptrWrapper : widgetPointers) { processWidget(*ptrWrapper); }


    return 0;
}

// Koan start
// Widget 1 (Gizmo) created.
// Processing via AbstractPtr: Widget ID: 1, Name: Gizmo
// (Raw pointer: 0x504000000010)
// Accessed via *: Widget ID 1
// Widget 2 (Gadget) created.
// Processing via AbstractPtr: Widget ID: 2, Name: Gadget
// (Raw pointer: 0x506000000030)
// Accessed via *: Widget+ ID 2
// Processing via AbstractPtr: Pointer is null
// Widget 3 (Thingamajig) created.
// Widget 4 (Doodad) created.
// Processing via AbstractPtr: Widget ID: 3, Name: Thingamajig
// (Raw pointer: 0x504000000050)
// Accessed via *: Widget ID 3
// Processing via AbstractPtr: Widget ID: 4, Name: Doodad
// (Raw pointer: 0x506000000090)
// Accessed via *: Widget ID 4
// Widget 3 (Thingamajig) destroyed.
// Widget 4 (Doodad) destroyed.
// Widget 2 (Gadget) destroyed.
// Widget 1 (Gizmo) destroyed.