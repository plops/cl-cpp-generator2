module;
export module Vector;
export class Vector {
public:
  Vector(int s) : elem(new double[s]), sz(s) {}
  double &operator[](int i) const { return elem[i]; }
  int size() const { return sz; }

private:
  double *elem;
  int sz;
};
export bool operator==(const Vector &v1, const Vector &v2) {
  if (!(v1.size() == v2.size())) {
    return false;
  }
  for (auto i = 0; i < v1.size(); i += 1) {
    if (!(v1[i] == v2[i])) {
      return false;
    }
  }
  return true;
}
