#include <boost/hana.hpp>
#include <iomanip>
#include <iostream>
#include <string>
namespace hana = boost::hana;
struct Person {
  BOOST_HANA_DEFINE_STRUCT(Person, (std::string, name), (unsigned short, age));
};

int main(int argc, char **argv) {
  auto john{Person({"John", 30})};
  hana::for_each(john, hana::fuse([&](auto name, auto member) {
                   std::cout << std::setw(8) << hana::to<char const *>(name)
                             << ": " << std::dec << std::setw(8) << member
                             << " @ " << std::hex << &member << "\n";
                 }));
  return 0;
}
