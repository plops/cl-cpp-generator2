//
// Created by martin on 4/5/25.
//

#include "Vec.h"

Vec::Vec(int n)
: v{n}
{
}

void Vec::insert(float f)
{
  v.push_back(f);
}

void Vec::resize(int n)
{
  v.resize(n);
}

int Vec::size()
{
  return v.size();
}

float& Vec::operator[](int idx)
{
  return v[idx];
}