#include <boost/hana/define_struct.hpp>
#include <string>
namespace hana = boost::hana;
struct Person {
  BOOST_HANA_DEFINE_STRUCT(Person, (std::string, name), (unsigned short, age));
};

int main(int argc, char **argv) {
  auto john{Person({"John", 30})};
  return 0;
}
