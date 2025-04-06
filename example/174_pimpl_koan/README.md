2017 Meyers Effective Modern C++

184

c++ 14 return type template <typename T> decltype(auto) fun(); p159
 move = rvalue_cast

move requests on const objects are silently transformed to copy
std::forward only sometimes casts

167 time func invocation, a lambda that can call any function


effective modern c++ p 154 pimpl with unique ptr, delete, move, and copy
unique_ptr requires extra definition hoops that are not required when the pimpl is defined using shared_ptr


custom deleter instead of ManagedPtr

p143
custom initializer list
auto ilist = {10,20};
std::make_shared<vector>(ilist)
