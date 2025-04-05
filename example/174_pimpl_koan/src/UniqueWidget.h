//
// Created by martin on 4/5/25.
//

#ifndef UNIQUEWIDGET_H
#define UNIQUEWIDGET_H

#include "IWidget.h"
#include <memory>
#include "IContainer.h"

class UniqueWidget : public IWidget
{
public:
    explicit UniqueWidget(int start, std::unique_ptr<IContainer> container);
    int add(int a, int b) override;
    ~UniqueWidget() override;
    UniqueWidget(const UniqueWidget& other);
    UniqueWidget(UniqueWidget&& other) noexcept;
    UniqueWidget& operator=(const UniqueWidget& other);
    UniqueWidget& operator=(UniqueWidget&& other) noexcept;
    void insert(float) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};


#endif //UNIQUEWIDGET_H
