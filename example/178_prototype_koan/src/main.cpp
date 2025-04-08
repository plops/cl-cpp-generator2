#include <iostream>
#include <ostream>
#include <string>
using namespace std;

class Address {
public:
    Address(const string& street, const string& city, int suite) :
        street(street),XXX
        city(city),
        suite(suite) {}

    virtual Address* clone() {
        return new Address{street, city, suite};
    }

    friend std::ostream& operator<<(std::ostream& os, const Address& obj) {
        return os
                << "street: " << obj.street
                << " city: " << obj.city
                << " suite: " << obj.suite;
    }

private:
    string street;
    string city;
    int suite;
};

class ExtendedAddress : public Address {
public:
    ExtendedAddress(const string& street, const string& city, int suite, const string& country,
            const string& postcode) :
        Address(street, city, suite),
        country(country),
        postcode(postcode) {}

    ExtendedAddress* clone() override {
        return new ExtendedAddress(*this);
    }

    friend std::ostream& operator<<(std::ostream& os, const ExtendedAddress& obj) {
        return os
                << static_cast<const Address&>(obj)
                << " country: " << obj.country
                << " postcode: " << obj.postcode;
    }

private:
    string country;
    string postcode;
};

int main(int argc, char* argv[])
{
    ExtendedAddress ea{"123 Bla","London", 123,"UK", "SG20UU"};

    cout << ea << endl;
    Address &a = ea;

    cout << a << endl;

    auto cloned = a.clone();

    cout << *cloned << endl;
    cout << *dynamic_cast<ExtendedAddress*>(cloned) << endl;

    return 0;
}
