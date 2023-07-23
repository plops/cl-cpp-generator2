#ifndef ARGEXCEPTION_H
#define ARGEXCEPTION_H

#include <exception>
#include <string>

class ArgException : public std::exception {
public:
    explicit ArgException(std::string msg);

    const char *what() const noexcept override;

private:
    std::string msg_;
};

#endif /* !ARGEXCEPTION_H */