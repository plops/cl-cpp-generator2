//
// Created by martin on 4/5/25.
//

#ifndef UNIQUEWIDGET_H
#define UNIQUEWIDGET_H

#include "IWidget.h"
#include <memory>

class UniqueWidget : public IWidget
{
public:
    explicit UniqueWidget(int start);
    int add(int a, int b) override;
    ~UniqueWidget() override;
    UniqueWidget(const UniqueWidget& other);
    UniqueWidget(UniqueWidget&& other) noexcept;
    UniqueWidget& operator=(const UniqueWidget& other);
    UniqueWidget& operator=(UniqueWidget&& other) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};


#endif //UNIQUEWIDGET_H
