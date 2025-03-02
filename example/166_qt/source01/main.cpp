#include <QApplication>
#include <QPushButton>

int main(int argc, char* argv[])
{
    auto a{QApplication(argc, argv)};
    auto button{QPushButton("Hello world!", nullptr)};
    button.resize(200, 100);
    button.show();
    return QApplication::exec();
}