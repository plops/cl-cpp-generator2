//
// Created by martin on 4/5/25.
//

#ifndef IWIDGET_H
#define IWIDGET_H

class IWidget
{
public:
  virtual ~IWidget() = default;
  virtual int add(int a, int b) = 0;
  virtual void insert(float) = 0;
};

#endif //IWIDGET_H
