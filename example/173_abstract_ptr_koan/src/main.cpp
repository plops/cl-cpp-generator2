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
    int    id;
    string name;

    Widget(int i, string n) :
        id{i}, name{move(n)} { cout << "Widget " << id << " (" << name << ") created." << endl; }
};
