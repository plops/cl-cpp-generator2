//
// Created by martin on 4/1/25.
//

#include <iostream>
#include <memory>
#include <string>
#include <cassert>

using namespace std;

class Widget {
public:
    Widget(int i, string n) :
        id{i}, name{move(n)} { cout << "Widget " << id << " (" << name << ") created." << endl; }

    virtual      ~Widget() { cout << "Widget " << id << " (" << name << ") destroyed." << endl; }
    virtual void display() const { cout << "Widget ID: " << id << ", Name: " << name << endl; }

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
