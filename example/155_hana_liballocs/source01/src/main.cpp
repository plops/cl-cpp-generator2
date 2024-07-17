#include <boost/hana.hpp>
#include <iostream>
#include <string>
namespace hana = boost::hana;
struct Person {
  BOOST_HANA_DEFINE_STRUCT(Person, (std::string, name), (unsigned short, age));
};

int main(int argc, char **argv) {
  auto john{Person({"John", 30})};
  hana::for_each(john, hana::fuse([&](auto name, auto member) {
                   std::cout << hana::to<char const *>(name) << ": " << member
                             << "\n";
                 }));
  return 0;
}
