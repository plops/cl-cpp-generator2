//
// Created by martin on 4/5/25.
//

#ifndef IWIDGET_H
#define IWIDGET_H

class IWidget {
public:
  ~IWidget() = default;
  virtual int add(int a, int b);
};

#endif //IWIDGET_H
